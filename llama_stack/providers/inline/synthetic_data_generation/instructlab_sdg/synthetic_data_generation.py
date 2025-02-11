# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.inference import Inference
from llama_stack.apis.pipelines import Pipelines
from llama_stack.apis.pipelines import Pipeline

from llama_stack.providers.datatypes import PipelinesProtocolPrivate
from llama_stack.providers.utils.kvstore import kvstore_impl

from .....apis.synthetic_data_generation.synthetic_data_generation import SyntheticDataGeneration, SyntheticDataGenerationResponse

from .config import InstructLabSDGConfig


PIPELINES_PREFIX = "instructlab_sdg_pipelines:"


class InstructLabSDGImpl(Pipelines, PipelinesProtocolPrivate):
    def __init__(
        self,
        config: InstructLabSDGConfig,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
        inference_api: Inference,
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api
        self.inference_api = inference_api
        # local registry for keeping track of pipelines within the provider
        self.pipeline_infos = {}
        self.kvstore = None

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.kvstore)
        # Load existing pipelines from kvstore
        start_key = PIPELINES_PREFIX
        end_key = f"{PIPELINES_PREFIX}\xff"
        stored_pipelines = await self.kvstore.range(start_key, end_key)

        for pipeline in stored_pipelines:
            pipeline = Pipeline.model_validate_json(pipeline)
            self.pipeline_infos[pipeline.identifier] = pipeline

    async def shutdown(self) -> None: ...

    async def register_pipeline(
        self,
        pipeline: Pipeline,
    ) -> None:
        # Store in kvstore
        key = f"{PIPELINES_PREFIX}{pipeline.identifier}"
        await self.kvstore.set(
            key=key,
            value=pipeline.json(),
        )
        self.pipeline_infos[pipeline.identifier] = pipeline
        print(f"!!! REGISTERED PIPELINE {pipeline}")

    async def unregister_pipeline(self, pipeline_id: str) -> None:
        key = f"{PIPELINES_PREFIX}{pipeline_id}"
        await self.kvstore.delete(key=key)
        del self.pipeline_infos[pipeline_id]
        print(f"!!! UNREGISTERED PIPELINE {pipeline_id}")

    async def synthetic_data_generate(
        self,
        dataset_id: str,
        pipeline_id: str,
    ) -> SyntheticDataGenerationResponse:
        print(f"!!! Running generate on dataset {dataset_id} with pipeline {pipeline_id}")
        pipeline = self.pipeline_infos[pipeline_id]
        print(f"!!! Found pipeline {pipeline}")
        # generated_data = await self._run_model_generation(provider_config, model)
        generated_data = []

        return SyntheticDataGenerationResponse(synthetic_data=generated_data, statistics={})
