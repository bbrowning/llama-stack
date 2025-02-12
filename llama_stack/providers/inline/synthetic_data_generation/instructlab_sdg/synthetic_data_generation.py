# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from typing import Any, Dict, List, Optional
import asyncio
import glob
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
        # print(f"!!! REGISTERED PIPELINE {pipeline}")

    async def unregister_pipeline(self, pipeline_id: str) -> None:
        key = f"{PIPELINES_PREFIX}{pipeline_id}"
        await self.kvstore.delete(key=key)
        del self.pipeline_infos[pipeline_id]
        # print(f"!!! UNREGISTERED PIPELINE {pipeline_id}")

    async def synthetic_data_generate(
        self,
        dataset_id: str,
        pipeline_id: str,
    ) -> SyntheticDataGenerationResponse:
        # print(f"!!! Running generate on dataset {dataset_id} with pipeline {pipeline_id}")
        pipeline = self.pipeline_infos[pipeline_id]
        # print(f"!!! Found pipeline {pipeline}")
        generated_data = await self._run_model_generation(dataset_id, pipeline)

        return SyntheticDataGenerationResponse(synthetic_data=generated_data, statistics={})

    async def _run_model_generation(
        self,
        dataset_id: str,
        pipeline: Pipeline,
    ) -> List[Dict[str, Any]]:
        # print(f"!!! dataset_id {dataset_id}")
        # print(f"!!! pipeline_id {pipeline.pipeline_id}")

        all_rows = await self.datasetio_api.get_rows_paginated(
            dataset_id=dataset_id,
            rows_in_page=-1,
        )
        # HACK: use a logger, but figure out how to set log levels as well
        print(f"Running InstructLab synthetic data generation on input dataset {dataset_id} with {len(all_rows.rows)} rows and pipeline {pipeline.identifier}.")

        # HACK: we won't have these env variables set when running as a real server
        base_url = os.getenv("OPENAI_ENDPOINT")
        assert base_url, "Set the environment variable OPENAI_ENDPOINT to your OpenAI Server endpoint"
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key, "Set the environment variable OPENAI_API_KEY to your OpenAI API key"
        client = OpenAI(base_url=base_url, api_key=api_key)
        client.server_supports_batched = True
        if all_rows.rows and all_rows.rows[0].get("qna_path"):
            # Entire e2e data generation
            return await asyncio.to_thread(
                self._ilab_data_generate,
                all_rows.rows,
                pipeline,
                client,
            )

        return await asyncio.to_thread(
            # Just running a single pipeline
            self._ilab_run_pipeline,
            all_rows.rows,
            pipeline,
            client,
        )

    def _ilab_run_pipeline(
        self,
        input_rows: [],
        pipeline: Pipeline,
        client,
    ) -> List[Dict[str, Any]]:
        ##
        ## Example of running only a single pipeline
        ##
        ## This expects input_rows to NOT be an entire taxonomy
        ##
        pipeline_context = IlabPipelineContext(
            client=client,
            model_family=pipeline.metadata.get("model_family", "mixtral"),
            model_id=pipeline.metadata.get("model_id"),
            num_instructions_to_generate=30,
        )
        temp_file_path = os.path.join(self.tempdir, f"{pipeline.pipeline_id}.yaml")
        with open(temp_file_path, "w", encoding="utf-8") as temp_file:
            temp_file.write(pipeline.metadata["pipeline_yaml"])
        ilab_pipeline = IlabPipeline.from_file(pipeline_context, temp_file_path)
        # HACK to just deal with 3 rows for testing expediency
        input_ds = Dataset.from_list(input_rows[0:3])
        output_ds = ilab_pipeline.generate(input_ds)
        return output_ds

    def _ilab_data_generate(
        self,
        input_rows: [],
        pipeline: Pipeline,
        client,
    ) -> List[Dict[str, Any]]:

        ##
        ## Example of running entire e2e ilab data generate
        ##
        ## This expects input_rows to contain the entire taxonomy
        ##
        date_suffix = (
            datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
        )
        taxonomy_dir = os.path.join(self.tempdir, "taxonomy")
        preprocessed_dir = os.path.join(self.tempdir, "preprocessed")
        generated_dir = os.path.join(self.tempdir, "generated")
        postprocessed_dir = os.path.join(self.tempdir, "postprocessed")
        # HACK handle if this isn't set? or make 1st-class param?
        sdg_pipeline = pipeline.metadata["pipeline"]
        # HACK handle if this isn't set? or make 1st-class param?
        teacher_model_path = pipeline.metadata["teacher_model_path"]
        # HACK handle if this isn't set? or make 1st-class param?
        model_id = pipeline.metadata["model_id"]

        # reconstruct our taxonomy on disk
        # HACK because this feels really hacky, and we need a better dataset
        # representation of a taxonomy
        for row in input_rows:
            qna_path = os.path.join(taxonomy_dir, row["qna_path"])
            qna_contents = row["qna_contents"]
            os.makedirs(os.path.dirname(qna_path), exist_ok=True)
            with open(qna_path, "w", encoding="utf-8") as qna_file:
                qna_file.write(qna_contents)

        # Now turn taxonomy into samples
        preprocess_taxonomy(
            taxonomy_dir=taxonomy_dir,
            output_dir=preprocessed_dir,
            teacher_model_path=teacher_model_path,
        )
        preprocessed_files = glob.glob(
            os.path.join(preprocessed_dir, "**", "*"),
            recursive=True,
        )
        # print(f"""!!! preprocessed_files\n{"\n".join(preprocessed_files)}""")

        # HACK hardcoded num_cpus, batch_size
        generate_taxonomy(
            client=client,
            input_dir=preprocessed_dir,
            output_dir=generated_dir,
            pipeline=sdg_pipeline,
            model_id=model_id,
            num_cpus=4,
            batch_size=8,
        )
        generated_files = glob.glob(
            os.path.join(generated_dir, "**", "*"),
            recursive=True,
        )
        # print(f"""!!! generated_files\n{"\n".join(generated_files)}""")

        postprocess_taxonomy(
            input_dir=generated_dir,
            output_dir=postprocessed_dir,
            date_suffix=date_suffix,
            pipeline=sdg_pipeline,
        )
        postprocessed_files = glob.glob(
            os.path.join(postprocessed_dir, "**", "*"),
            recursive=True,
        )
        # print(f"""!!! postprocessed_files\n{"\n".join(postprocessed_files)}""")

        mixed_skills_output_file = f"{postprocessed_dir}/skills_train_msgs_{date_suffix}.jsonl"
        mix_datasets(
            recipe_file=f"{postprocessed_dir}/skills_recipe_{date_suffix}.yaml",
            output_file=mixed_skills_output_file,
        )

        mixed_knowledge_output_file = f"{postprocessed_dir}/knowledge_train_msgs_{date_suffix}.jsonl"
        mix_datasets(
            recipe_file=f"{postprocessed_dir}/knowledge_recipe_{date_suffix}.yaml",
            output_file=mixed_knowledge_output_file,
        )

        # HACK need a better way to output multiple datasets here
        # or a single dataset with all the needed data in it and some splits?
        output_files = [
            {
                "type": "mixed_skills",
                "path": mixed_skills_output_file,
            },
            {
                "type": "mixed_knowledge",
                "path": mixed_knowledge_output_file,
            }
        ]
        return output_files
