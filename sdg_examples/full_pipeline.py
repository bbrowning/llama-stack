import os

from llama_stack import LlamaStackAsLibraryClient
from llama_stack.apis.common.content_types import URL

model_id = os.getenv("OPENAI_MODEL_ID")
assert model_id, "Set the environment variable OPENAI_MODEL_ID to your OpenAI model's id"

teacher_model_path = os.getenv("TEACHER_MODEL_PATH")
assert teacher_model_path, "Set the environment variable TEACHER_MODEL_PATH to your teacher model's location on disk"
teacher_model_path = os.path.expanduser(teacher_model_path)

client = LlamaStackAsLibraryClient("ollama")
client.initialize()


#
# An example of running the `ilab data generate --pipeline full` flow in
# Llama Stack.
#
taxonomy_dataset_id = "rhelai-sample-taxonomy"
client.datasets.register(
    dataset_id=taxonomy_dataset_id,
    provider_id="instructlab-taxonomy",
    url={"uri": "https://github.com/RedHatOfficial/rhelai-sample-taxonomy.git"},
    metadata={
        # relevant metadata about this taxonomy
    },
    dataset_schema={
        # omitting schema here for brevity
    },
)
full_pipeline_id = "instructlab-full"
client.pipelines.register(
    pipeline_id=full_pipeline_id,
    provider_id="instructlab-sdg",
    input_dataset_schema={
        # omitting schema here for brevity
    },
    metadata={
        # provider-specific metadata about this pipeline
        "pipeline": "full",  # for a built-in e2e pipeline
        "teacher_model_path": teacher_model_path,
        "model_id": model_id,
    },
)

generate_results = client.synthetic_data_generation.generate(
    dataset_id=taxonomy_dataset_id,
    pipeline_id=full_pipeline_id,
)
print(f"!!! results {generate_results}")

client.pipelines.unregister(pipeline_id=full_pipeline_id)
