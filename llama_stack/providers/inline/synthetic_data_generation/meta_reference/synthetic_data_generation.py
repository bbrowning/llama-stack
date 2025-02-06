# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from datetime import datetime
import os
import shutil
import tempfile

from llama_stack.apis.agents import Agents, StepType
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.inference import Inference, Message, UserMessage
from llama_stack.apis.scoring import Scoring
from llama_stack.distribution.datatypes import Api

from llama_stack.providers.utils.common.data_schema_validator import (
    ColumnName,
    get_valid_schemas,
    validate_dataset_schema,
)

from .....apis.synthetic_data_generation.synthetic_data_generation import FilteringFunction, SyntheticDataGeneration, SyntheticDataGenerationResponse

from .config import MetaReferenceSyntheticDataGenerationConfig

from instructlab.sdg.generate_data import (
    generate_taxonomy,
    mix_datasets,
    postprocess_taxonomy,
    preprocess_taxonomy,
)
from openai import OpenAI


class MetaReferenceSyntheticDataGenerationImpl(
    SyntheticDataGeneration,
):
    def __init__(
        self,
        config: MetaReferenceSyntheticDataGenerationConfig,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
        scoring_api: Scoring,
        inference_api: Inference,
        agents_api: Agents,
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api
        self.scoring_api = scoring_api
        self.inference_api = inference_api
        self.agents_api = agents_api

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def _run_model_generation(
        self,
        provider_config: Dict[str, Any],
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        print(f"!!! provider_config {provider_config}")
        print(f"!!! model {model}")

        client_taxonomy_path = provider_config["taxonomy_dir"]
        pipeline = provider_config["pipeline"]
        teacher_model_path = provider_config["teacher_model_path"]

        with tempfile.TemporaryDirectory() as temp_dir:
            taxonomy_dir = f"{temp_dir}/taxonomy"
            shutil.copytree(client_taxonomy_path, taxonomy_dir)
            return await self._ilab_data_generate(
                taxonomy_dir=taxonomy_dir,
                pipeline=pipeline,
                teacher_model_path=teacher_model_path,
                model=model,
            )

    async def _ilab_data_generate(
            self,
            taxonomy_dir: str,
            pipeline: str,
            teacher_model_path: str,
            model: str,
    ) -> List[Dict[str, Any]]:
        # for x in tqdm(input_rows):
        #     response = await self.inference_api.chat_completion(
        #         model_id=os.environ["INFERENCE_MODEL"],
        #         messages=[x],
        #     )
        #     generations.append({
        #         "prompt": x.content,
        #         "response": response.completion_message.content
        #     })

        date_suffix = (
            datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
        )
        preprocessed_dir = "preprocessed"
        generated_dir = "generated"
        postprocessed_dir = "postprocessed"

        client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        client.server_supports_batched = True

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

        postprocess_taxonomy(
            input_dir=generated_dir,
            output_dir=postprocessed_dir,
            date_suffix=date_suffix,
            pipeline=pipeline,
        )

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

    async def synthetic_data_generate(
        self,
        provider_config: Dict[str, Any],
        model: Optional[str] = None,
    ) -> SyntheticDataGenerationResponse:
        generated_data = await self._run_model_generation(provider_config, model)

        return SyntheticDataGenerationResponse(synthetic_data=generated_data, statistics={})
