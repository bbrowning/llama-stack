
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Literal, Optional, Protocol

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel, Field

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.common.type_system import ParamType
from llama_stack.apis.resource import Resource, ResourceType


class CommonPipelineFields(BaseModel):
    input_dataset_schema: Dict[str, ParamType]
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this pipeline",
    )


@json_schema_type
class Pipeline(CommonPipelineFields, Resource):
    type: Literal[ResourceType.pipeline.value] = ResourceType.pipeline.value

    @property
    def pipeline_id(self) -> str:
        return self.identifier

    @property
    def provider_pipeline_id(self) -> str:
        return self.provider_resource_id


class PipelineInput(CommonPipelineFields, BaseModel):
    pipeline_id: str
    provider_id: Optional[str] = None
    provider_pipeline_id: Optional[str] = None


class ListPipelinesResponse(BaseModel):
    data: List[Pipeline]


class Pipelines(Protocol):
    @webmethod(route="/pipelines", method="POST")
    async def register_pipeline(
        self,
        pipeline_id: str,
        input_dataset_schema: Dict[str, ParamType],
        provider_pipeline_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    @webmethod(route="/pipelines/{pipeline_id}", method="GET")
    async def get_pipeline(
        self,
        pipeline_id: str,
    ) -> Optional[Pipeline]: ...

    @webmethod(route="/pipelines", method="GET")
    async def list_pipelines(self) -> ListPipelinesResponse: ...

    @webmethod(route="/pipelines/{pipeline_id}", method="DELETE")
    async def unregister_pipeline(
        self,
        pipeline_id: str,
    ) -> None: ...
