# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

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


@pytest.fixture
def sdg_config(sqlite_kvstore):
    return InstructLabSDGConfig(kvstore=sqlite_kvstore)


@pytest.fixture
def mock_datasetio_api():
    mock = MagicMock()
    mock_data = MagicMock()
    mock_data.data = [
        {"output": "foo"},
        {"output": "baz"},
    ]
    mock.iterrows = AsyncMock(return_value=mock_data)
    return mock


@pytest.fixture
def mock_inference_api():

    completion_result = MagicMock(name="mock_completion_result")
    completion_result.choices = []
    completion_result.content = "foo"
    completion_result.stop_reason = "end_of_turn"

    mock = MagicMock()
    mock.completion = MagicMock(return_value=completion_result)

    return mock


@pytest_asyncio.fixture
async def sdg_impl(sdg_config, mock_datasetio_api, mock_inference_api):
    sdg = InstructLabSDGImpl(sdg_config, mock_datasetio_api, None, mock_inference_api, None)
    await sdg.initialize()
    return sdg


@pytest.mark.asyncio
async def test_manage_sdg_functions(sdg_impl):
    sdg_fn = SDGFn(identifier="foo", provider_resource_id="foo", provider_id="instructlab-sdg")

    await sdg_impl.register_sdg_function(sdg_fn)
    sdg_fns = await sdg_impl.list_sdg_functions()
    assert sdg_fns
    assert len(sdg_fns) == 1

    await sdg_impl.unregister_sdg_function(sdg_fn.identifier)
    sdg_fns = await sdg_impl.list_sdg_functions()
    assert len(sdg_fns) == 0


@pytest.fixture
def duplicate_columns_pipeline_yaml():
    return """
- block_type: DuplicateColumns
  block_config:
    block_name: duplicate_document_col
    columns_map:
      output: original_output
    """


@pytest.fixture
def llmblock_contents():
    return """
- block_type: LLMBlock
  block_config:
    block_name: generate
    model_id: foo
    config_path: foo_llmblock.yaml
    output_cols:
    - response
    """


@pytest.fixture
def llmblock_config():
    return """
system: ~
introduction: ~
principles: ~
examples: ~
generation: ~
start_tags: [""]
end_tags: [""]
    """


@pytest.mark.asyncio
async def test_generate_basic(sdg_impl, duplicate_columns_pipeline_yaml):
    sdg_fn = SDGFn(
        identifier="foo",
        provider_resource_id="foo",
        provider_id="instructlab-sdg",
        params={
            "type": "instructlab_sdg",
            "pipeline_yaml": duplicate_columns_pipeline_yaml,
        },
    )

    await sdg_impl.register_sdg_function(sdg_fn)

    response = await sdg_impl.synthetic_data_generate("my-dataset", "foo")
    assert response
    assert response.synthetic_data
    assert len(response.synthetic_data) == 2
    assert response.synthetic_data[0].get("original_output", None)


@pytest.mark.asyncio
async def test_generate_llmblock(sdg_impl, llmblock_contents, llmblock_config):
    extra_configs = {
        "foo_llmblock.yaml": llmblock_config,
    }

    sdg_fn = SDGFn(
        identifier="foo",
        provider_resource_id="foo",
        provider_id="instructlab-sdg",
        params={
            "type": "instructlab_sdg",
            "pipeline_yaml": llmblock_contents,
            "extra_configs": extra_configs,
            "chat_templates": {
                "foo": "asdf",
            },
        },
    )

    await sdg_impl.register_sdg_function(sdg_fn)

    response = await sdg_impl.synthetic_data_generate(
        dataset_id="my-dataset",
        sdg_fn_id="foo",
        config={
            "num_workers": 2,
            "batch_size": 16,
        },
    )
    assert response
    assert response.synthetic_data
    assert len(response.synthetic_data) == 2
    assert response.synthetic_data[0].get("response", None) == "foo"
