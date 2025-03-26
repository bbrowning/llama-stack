# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os
import tempfile
from typing import Any, Dict, List, Optional

from datasets import Dataset as HFDataset
from instructlab.sdg.flow import Flow
from instructlab.sdg.pipeline import Pipeline
from instructlab.sdg.registry import PromptRegistry
from instructlab.sdg.sdg import SDG
from lls_openai_client.client_adapter import OpenAIClientAdapter

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.inference import Inference
from llama_stack.apis.models import Models
from llama_stack.apis.sdg_functions import InstructLabSDGFnParams, SDGFn, SDGFunctions
from llama_stack.providers.datatypes import SDGFunctionsProtocolPrivate
from llama_stack.providers.utils.kvstore import kvstore_impl

from .....apis.synthetic_data_generation.synthetic_data_generation import (
    GenerateConfig,
    SyntheticDataGenerationResponse,
)
from .config import InstructLabSDGConfig
from .server_client import ServerLlamaStackClient

SDG_FNS_PREFIX = "instructlab_sdg_functions:"


class InstructLabSDGImpl(SDGFunctions, SDGFunctionsProtocolPrivate):
    def __init__(
        self,
        config: InstructLabSDGConfig,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
        inference_api: Inference,
        models_api: Models,
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api
        self.inference_api = inference_api
        self.models_api = models_api
        self.sdg_fns = {}
        self.kvstore = None

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.kvstore)
        # Load existing pipelines from kvstore
        start_key = SDG_FNS_PREFIX
        end_key = f"{SDG_FNS_PREFIX}\xff"
        stored_sdg_fns = await self.kvstore.range(start_key, end_key)

        for sdg_fn in stored_sdg_fns:
            sdg_fn = SDGFn.model_validate_json(sdg_fn)
            self.sdg_fns[sdg_fn.identifier] = sdg_fn

    async def shutdown(self) -> None: ...

    async def list_sdg_functions(self) -> List[SDGFn]:
        return list(self.sdg_fns.values())

    async def register_sdg_function(self, sdg_fn: SDGFn) -> None:
        # Store in kvstore
        key = f"{SDG_FNS_PREFIX}{sdg_fn.identifier}"
        await self.kvstore.set(
            key=key,
            value=sdg_fn.model_dump_json(),
        )
        self.sdg_fns[sdg_fn.identifier] = sdg_fn

    async def unregister_sdg_function(self, sdg_fn_id: str) -> None:
        key = f"{SDG_FNS_PREFIX}{sdg_fn_id}"
        await self.kvstore.delete(key=key)
        del self.sdg_fns[sdg_fn_id]

    async def synthetic_data_generate(
        self,
        dataset_id: str,
        sdg_fn_id: str,
        config: Optional[GenerateConfig] = None,
    ) -> SyntheticDataGenerationResponse:
        print(f"!!! Running generate on dataset {dataset_id} with sdg_fn {sdg_fn_id}")
        sdg_fn = self.sdg_fns[sdg_fn_id]
        print(f"!!! Found sdg_fn {sdg_fn}")
        if not sdg_fn:
            raise ValueError(f"SDG function {sdg_fn_id} is not registered with this provider.")

        params = sdg_fn.params
        if not isinstance(params, InstructLabSDGFnParams):
            raise ValueError(f"SDG function {sdg_fn_id} should use valid InstructLabSDGFnParams.")

        input_rows = await self.datasetio_api.get_rows_paginated(
            dataset_id=dataset_id,
            rows_in_page=-1,
        )
        if not input_rows.rows:
            raise ValueError(f"Dataset {dataset_id} contains no rows.")
        input_rows = input_rows.rows

        input_rows = [{"output": "foo"}, {"output": "bar"}]
        print(f"!!! input_rows {input_rows}")
        # Convert the list of input rows to a Dataset
        input_ds = HFDataset.from_list(input_rows)

        pipeline_yaml = params.pipeline_yaml
        extra_configs = params.extra_configs
        chat_templates = params.chat_templates
        server_client = ServerLlamaStackClient(self.inference_api, self.models_api)
        client = OpenAIClientAdapter(server_client)
        output_ds = await asyncio.to_thread(
            self._run_generate,
            client,
            input_ds,
            pipeline_yaml,
            extra_configs,
            chat_templates,
        )

        # Convert the Dataset to a list, which is risky for large datasets fitting in memory...
        output_rows = output_ds.to_list()
        print(f"!!! output_rows {output_rows}")

        return SyntheticDataGenerationResponse(synthetic_data=output_rows, statistics={})


    def _run_generate(
        self,
        client: OpenAIClientAdapter,
        input_ds: HFDataset,
        pipeline_yaml: str,
        extra_configs: Dict[str, Any],
        chat_templates: Dict[str, str],
    ) -> HFDataset:
        pipeline_yaml_name = "pipeline.yaml"
        config_files = {
            pipeline_yaml_name: pipeline_yaml,
        }
        if extra_configs:
            config_files.update(extra_configs)

        self._register_chat_templates(chat_templates)

        num_workers = 2
        batch_size = 8
        save_freq = 2

        with tempfile.TemporaryDirectory() as temp_dir:
            for config_file_name, contents in config_files.items():
                with open(os.path.join(temp_dir, config_file_name), "w", encoding="utf-8") as config_file:
                    config_file.write(contents)

            flow = Flow(client)
            # Set the flow's base_path so relative links resolve properly
            flow.base_path = temp_dir
            flow_cfg = flow.get_flow_from_file(os.path.join(temp_dir, pipeline_yaml_name))
            sdg = SDG(
                [Pipeline(flow_cfg)],
                num_workers=num_workers,
                batch_size=batch_size,
                save_freq=save_freq,
            )
            return sdg.generate(input_ds)


    def _register_chat_templates(
        self,
        chat_templates: Dict[str, str],
    ):
        if chat_templates:
            for model_name, chat_template in chat_templates.items():
                PromptRegistry.register(model_name)(lambda: chat_template)
