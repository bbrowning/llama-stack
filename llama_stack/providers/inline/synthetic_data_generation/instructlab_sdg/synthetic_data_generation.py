# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Optional

from datetime import datetime
import asyncio
import os
import tempfile

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.inference import Inference
from llama_stack.apis.pipelines import Pipelines
from llama_stack.apis.pipelines import Pipeline

from llama_stack.providers.datatypes import PipelinesProtocolPrivate
from llama_stack.providers.utils.kvstore import kvstore_impl

from .....apis.synthetic_data_generation.synthetic_data_generation import SyntheticDataGeneration, SyntheticDataGenerationResponse

from .config import InstructLabSDGConfig

from datasets import Dataset
from openai import OpenAI

from instructlab.sdg.generate_data import (
    _sdg_init,
    generate_taxonomy,
    mix_datasets,
    postprocess_taxonomy,
    preprocess_taxonomy,
)
from instructlab.sdg.pipeline import Pipeline as IlabPipeline, PipelineContext as IlabPipelineContext


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
        self.tempdir = tempfile.mkdtemp()

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
        generated_data = await self._run_model_generation(dataset_id, pipeline)

        return SyntheticDataGenerationResponse(synthetic_data=generated_data, statistics={})

    async def _run_model_generation(
        self,
        dataset_id: str,
        pipeline: Pipeline,
    ) -> List[Dict[str, Any]]:
        print(f"!!! dataset_id {dataset_id}")
        print(f"!!! pipeline_id {pipeline.pipeline_id}")

        all_rows = await self.datasetio_api.get_rows_paginated(
            dataset_id=dataset_id,
            rows_in_page=-1,
        )
        print(f"!!! input dataset has {len(all_rows.rows)} rows")

        return await asyncio.to_thread(self._ilab_data_generate, all_rows.rows, pipeline)

        # return await self._ilab_data_generate(
        # )
        # client_taxonomy_path = provider_config["taxonomy_dir"]
        # pipeline = provider_config["pipeline"]
        # teacher_model_path = provider_config["teacher_model_path"]

        # with tempfile.TemporaryDirectory() as temp_dir:
        #     taxonomy_dir = f"{temp_dir}/taxonomy"
        #     shutil.copytree(client_taxonomy_path, taxonomy_dir)
        #     return await self._ilab_data_generate(
        #         taxonomy_dir=taxonomy_dir,
        #         pipeline=pipeline,
        #         teacher_model_path=teacher_model_path,
        #         model=model,
        #     )

    def _ilab_data_generate(
            self,
            input_rows: [],
            pipeline: Pipeline,
    ) -> List[Dict[str, Any]]:
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        client.server_supports_batched = True
        pipeline_context = IlabPipelineContext(
            client=client,
            model_family="mixtral",
            model_id="mixtral-foo",
            num_instructions_to_generate=30,
        )

        temp_file_path = os.path.join(self.tempdir, f"{pipeline.pipeline_id}.yaml")
        with open(temp_file_path, "w", encoding="utf-8") as temp_file:
            temp_file.write(pipeline.metadata["pipeline_yaml"])
        ilab_pipeline = IlabPipeline.from_file(pipeline_context, temp_file_path)
        # HACK to just deal with 3 rows for testing
        input_ds = Dataset.from_list(input_rows[0:3])
        output_ds = ilab_pipeline.generate(input_ds, "foo")
        return output_ds

        # date_suffix = (
        #     datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
        # )
        # preprocessed_dir = "preprocessed"
        # generated_dir = "generated"
        # postprocessed_dir = "postprocessed"

        # client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        # client.server_supports_batched = True

        # preprocess_taxonomy(
        #     taxonomy_dir=taxonomy_dir,
        #     output_dir=preprocessed_dir,
        #     teacher_model_path=teacher_model_path,
        # )

        # generate_taxonomy(
        #     client=client,
        #     input_dir=preprocessed_dir,
        #     output_dir=generated_dir,
        #     pipeline=pipeline,
        #     model_id=model,
        #     num_cpus=4,
        #     batch_size=8,
        # )

        # postprocess_taxonomy(
        #     input_dir=generated_dir,
        #     output_dir=postprocessed_dir,
        #     date_suffix=date_suffix,
        #     pipeline=pipeline,
        # )

        # mixed_skills_output_file = f"{postprocessed_dir}/skills_train_msgs_{date_suffix}.jsonl"
        # mix_datasets(
        #     recipe_file=f"{postprocessed_dir}/skills_recipe_{date_suffix}.yaml",
        #     output_file=mixed_skills_output_file,
        # )

        # mixed_knowledge_output_file = f"{postprocessed_dir}/knowledge_train_msgs_{date_suffix}.jsonl"
        # mix_datasets(
        #     recipe_file=f"{postprocessed_dir}/knowledge_recipe_{date_suffix}.yaml",
        #     output_file=mixed_knowledge_output_file,
        # )

        # output_files = [
        #     {
        #         "type": "mixed_skills",
        #         "path": mixed_skills_output_file,
        #     },
        #     {
        #         "type": "mixed_knowledge",
        #         "path": mixed_knowledge_output_file,
        #     }
        # ]
        # return output_files
