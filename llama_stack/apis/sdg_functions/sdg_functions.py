# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod


@json_schema_type
class SDGFnParamsType(Enum):
    instructlab_sdg = "instructlab_sdg"


@json_schema_type
class InstructLabSDGFnParams(BaseModel):
    type: Literal[SDGFnParamsType.instructlab_sdg.value] = SDGFnParamsType.instructlab_sdg.value
    pipeline_yaml: Optional[str] = None
    extra_configs: Dict[str, Any] = None
    chat_templates: Dict[str, str] = None


SDGFnParams = register_schema(
    Annotated[
        Union[
            InstructLabSDGFnParams,
            # Other SDG providers may have different params
        ],
        Field(discriminator="type"),
    ],
    name="SDGFnParams",
)


class CommonSDGFnFields(BaseModel):
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this definition",
    )
    params: Optional[SDGFnParams] = Field(
        description="The parameters for the SDG function",
        default=None,
    )


@json_schema_type
class SDGFn(CommonSDGFnFields, Resource):
    type: Literal[ResourceType.sdg_function.value] = ResourceType.sdg_function.value

    @property
    def sdg_fn_id(self) -> str:
        return self.identifier

    @property
    def provider_sdg_fn_id(self) -> str:
        return self.provider_resource_id


class SDGFnInput(CommonSDGFnFields, BaseModel):
    sdg_fn_id: str
    provider_id: Optional[str] = None
    provider_sdg_fn_id: Optional[str] = None


class ListSDGFunctionsResponse(BaseModel):
    data: List[SDGFn]


class SDGFunctions(Protocol):
    @webmethod(route="/sdg_functions", method="GET")
    async def list_sdg_functions(self) -> ListSDGFunctionsResponse: ...

    @webmethod(route="/sdg_functions/{sdg_fn_id}", method="GET")
    async def get_sdg_function(
        self,
        sdg_fn_id: str,
    ) -> SDGFn: ...

    @webmethod(route="/sdg_functions", method="POST")
    async def register_sdg_function(
        self,
        sdg_fn_id: str,
        description: str,
        provider_sdg_fn_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        params: Optional[SDGFnParams] = None,
    ) -> None: ...

    @webmethod(route="/sdg_functions/{sdg_fn_id}", method="DELETE")
    async def unregister_sdg_function(
        self,
        sdg_fn_id: str,
    ) -> None: ...
