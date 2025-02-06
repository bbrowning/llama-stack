# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any, Dict, List, Optional

from tqdm import tqdm

import os

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

from datetime import datetime
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
    ) -> List[Dict[str, Any]]:
        generations = []
        # for x in tqdm(input_rows):
        #     response = await self.inference_api.chat_completion(
        #         model_id=os.environ["INFERENCE_MODEL"],
        #         messages=[x],
        #     )
        #     generations.append({
        #         "prompt": x.content,
        #         "response": response.completion_message.content
        #     })

        pipeline_dir="/home/bbrownin/src/instructlab/sdg/src/instructlab/sdg/pipelines/full"
        date_suffix = (
            datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
        )
        preprocessed_dir = "preprocessed"
        generated_dir = "generated"
        postprocessed_dir = "postprocessed"

        # client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        # client.server_supports_batched = True

        # import logging
        # logging.basicConfig(
        #     level="DEBUG",
        #     format="%(levelname)s %(asctime)s %(name)s:%(lineno)d: %(message)s",
        # )

        # preprocess_taxonomy(
        #     taxonomy_dir="/home/bbrownin/src/instructlab/rhelai-sample-taxonomy",
        #     output_dir=preprocessed_dir,
        #     teacher_model_path="/home/bbrownin/src/instructlab/sdg/tests/testdata/models/instructlab/granite-7b-lab",
        # )

        # generate_taxonomy(
        #     client=client,
        #     input_dir=preprocessed_dir,
        #     output_dir=generated_dir,
        #     pipeline=pipeline_dir,
        #     model_id="/home/ec2-user/.cache/instructlab/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
        #     num_cpus=4,
        #     batch_size=8,
        # )

        postprocess_taxonomy(
            input_dir=generated_dir,
            output_dir=postprocessed_dir,
            date_suffix=date_suffix,
            pipeline=pipeline_dir,
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

        return generations

    async def synthetic_data_generate(
        self,
        dialogs: List[Message],
        filtering_function: FilteringFunction = FilteringFunction.none,
        model: Optional[str] = None,
    ) -> SyntheticDataGenerationResponse:
        generated_data = await self._run_model_generation(dialogs)

        return SyntheticDataGenerationResponse(synthetic_data=generated_data, statistics={})
