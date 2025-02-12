from llama_stack import LlamaStackAsLibraryClient
from llama_stack.apis.common.content_types import URL

client = LlamaStackAsLibraryClient("ollama")
client.initialize()

#
# An example of running a single SDG pipeline in Llama Stack using
# the alpaca dataset.
#
alpaca_dataset_id = "alpaca"
client.datasets.register(
    dataset_id=alpaca_dataset_id,
    provider_id="huggingface",
    url=URL(uri="https://huggingface.co/datasets/tatsu-lab/alpaca"),
    metadata={
        "path": "tatsu-lab/alpaca",
        "split": "train",
    },
    dataset_schema={
    },
)
pipeline_yaml = """
version: "1.0"
blocks:
  - name: duplicate_document_col
    type: DuplicateColumnsBlock
    config:
      columns_map:
        output: original_output
"""
custom_pipeline_id = "example_custom_pipeline"
client.pipelines.register(
    pipeline_id=custom_pipeline_id,
    provider_id="instructlab-sdg",
    input_dataset_schema={
        # omitting schema here for brevity
    },
    metadata={
        "pipeline_yaml": pipeline_yaml,
    },
)

generate_results = client.synthetic_data_generation.generate(
    dataset_id=alpaca_dataset_id,
    pipeline_id=custom_pipeline_id,
)
print(f"!!! results {generate_results}")

client.pipelines.unregister(pipeline_id=custom_pipeline_id)
