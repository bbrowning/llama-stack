from llama_stack import LlamaStackAsLibraryClient
from llama_stack.apis.common.content_types import URL
from llama_stack.apis.models import ModelType

client = LlamaStackAsLibraryClient("remote-vllm")
client.initialize()

client.models.register(
    model_id="mixtral-8x7b",
    provider_id="vllm-inference",
    provider_model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_type=ModelType.llm,
    metadata={
        "vllm_url": "http://localhost:8000/v1",
        "vllm_api_token": "EMPTY",
    }
)

client.models.register(
    model_id="merlinite-7b-lab",
    provider_id="vllm-inference",
    provider_model_id="instructlab/merlinite-7b-lab",
    model_type=ModelType.llm,
    metadata={
        "vllm_url": "http://localhost:8001/v1",
        "vllm_api_token": "EMPTY",
    }
)
