# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

import pytest

from llama_stack.apis.sdg_functions import SDGFn
from llama_stack.providers.inline.synthetic_data_generation.instructlab_sdg.config import InstructLabSDGConfig
from llama_stack.providers.inline.synthetic_data_generation.instructlab_sdg.synthetic_data_generation import (
    InstructLabSDGImpl,
)
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig

# These are unit test for the InstructLab synthetic data generation
# provider implementation.


@pytest.fixture
def sqlite_kvstore(tmp_path):
    return SqliteKVStoreConfig(db_path=str(tmp_path / "kvstore.db"))


@pytest.mark.asyncio
async def test_manage_sdg_functions(sqlite_kvstore):
    config = InstructLabSDGConfig(kvstore=sqlite_kvstore)
    sdg = InstructLabSDGImpl(config, None, None, None, None)
    await sdg.initialize()

    sdg_fn = SDGFn(identifier="foo", provider_resource_id="foo", provider_id="instructlab-sdg")

    await sdg.register_sdg_function(sdg_fn)
    sdg_fns = await sdg.list_sdg_functions()
    assert sdg_fns
    assert len(sdg_fns) == 1

    await sdg.unregister_sdg_function(sdg_fn.identifier)
    sdg_fns = await sdg.list_sdg_functions()
    assert len(sdg_fns) == 0


pipeline_yaml = """
- block_type: DuplicateColumns
  block_config:
    block_name: duplicate_document_col
    columns_map:
      output: original_output
"""


@pytest.mark.asyncio
async def test_generate(sqlite_kvstore):
    config = InstructLabSDGConfig(kvstore=sqlite_kvstore)
    mock_datasetio_api = AsyncMock()
    sdg = InstructLabSDGImpl(config, mock_datasetio_api, None, None, None)
    await sdg.initialize()

    sdg_fn = SDGFn(
        identifier="foo",
        provider_resource_id="foo",
        provider_id="instructlab-sdg",
        params={
            "type": "instructlab_sdg",
            "pipeline_yaml": pipeline_yaml,
        },
    )

    await sdg.register_sdg_function(sdg_fn)

    response = await sdg.synthetic_data_generate("my-dataset", "foo")
    assert response
    assert response.synthetic_data
    assert len(response.synthetic_data) == 2
    assert response.synthetic_data[0].get("original_output", None)
