# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Optional, Protocol, Union

from pydantic import BaseModel

from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class SyntheticDataGenerationRequest(BaseModel):
    """Request to generate synthetic data."""

    dataset_id: str
    sdg_fn_id: str


@json_schema_type
class SyntheticDataGenerationResponse(BaseModel):
    """Response from the synthetic data generation. Batch of (prompt, response, score) tuples that pass the threshold."""

    synthetic_data: List[Dict[str, Any]]
    statistics: Optional[Dict[str, Any]] = None


class SyntheticDataGeneration(Protocol):
    @webmethod(route="/synthetic-data-generation/generate")
    def synthetic_data_generate(
        self,
        dataset_id: str,
        sdg_fn_id: str,
    ) -> Union[SyntheticDataGenerationResponse]: ...
