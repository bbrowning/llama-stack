# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time

import httpx
import pytest
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.shared_params.sampling_params import (
    SamplingParams,
    StrategyTopPSamplingStrategy,
)
from openai.types.completion import Completion as OpenAICompletion
from openai.types.completion_choice import CompletionChoice as OpenAICompletionChoice


class Completions:
    def __init__(self, llama_stack_client):
        self.lls_client = llama_stack_client

    def create(self, *args, **kwargs):
        print(f"!!! calling completions.create with {args} and {kwargs}")
        model_id = kwargs["model"]
        content = kwargs["prompt"]
        sampling_params = SamplingParams()
        n = kwargs.get("n", 1)
        max_tokens = kwargs.get("max_tokens", None)
        if max_tokens:
            sampling_params["max_tokens"] = max_tokens
        temperature = kwargs.get("temperature", None)
        if temperature:
            top_p_sampling_strategy = StrategyTopPSamplingStrategy(
                type="top_p",
                temperature=temperature,
                top_p=1.0,
            )
            sampling_params["strategy"] = top_p_sampling_strategy
        choices = []
        for i in range(0, n):
            lls_result = self.lls_client.inference.completion(
                model_id=model_id,
                content=content,
                sampling_params=sampling_params,
            )
            stop_reason_map = {
                "end_of_turn": "stop",
                "end_of_message": "stop",
                "out_of_tokens": "length",
            }
            choice = OpenAICompletionChoice(
                index=i,
                text=lls_result.content,
                finish_reason=stop_reason_map[lls_result.stop_reason],
            )
            choices.append(choice)
        return OpenAICompletion(
            id="foo",
            choices=choices,
            created=int(time.time()),
            model=model_id,
            object="text_completion",
        )


class ChatCompletions:
    def __init__(self, llama_stack_client):
        self.lls_client = llama_stack_client

    def create(self, *args, **kwargs):
        print(f"!!! calling completions.create with {args} and {kwargs}")
        return OpenAICompletion(
            id="foo",
            choices=[],
            created=0,
            model="foo",
            object="text_completion",
        )


class Chat:
    completions: ChatCompletions

    def __init__(self, llama_stack_client):
        self.lls_client = llama_stack_client
        self.completions = ChatCompletions(self.lls_client)


class Models:
    def __init__(self, llama_stack_client):
        self.lls_client = llama_stack_client

    def list(self, *args, **kwargs):
        return self.lls_client.models.list()


class OpenAIClientAdapter:
    completions: Completions
    chat: Chat

    def __init__(self, llama_stack_client):
        self.lls_client = llama_stack_client
        if not self.lls_client:
            raise ValueError("A `llama_stack_client` must be provided.")

        self.completions = Completions(self.lls_client)
        self.chat = Chat(self.lls_client)
        self.models = Models(self.lls_client)

    @property
    def base_url(self) -> httpx.URL:
        return self.lls_client.base_url

    def get(self, *args, **kwargs):
        return self.lls_client.get(*args, **kwargs)


@pytest.fixture
def llama_stack_client():
    return LlamaStackClient(base_url="http://localhost:8321")


@pytest.fixture
def default_model_id(llama_stack_client):
    for model in llama_stack_client.models.list():
        if model.api_model_type == "llm":
            return model.provider_resource_id
    raise ValueError("No inference model available for testing")


def test_client_attributes(llama_stack_client):
    client = OpenAIClientAdapter(llama_stack_client)
    assert client.base_url


def test_client_get(llama_stack_client):
    client = OpenAIClientAdapter(llama_stack_client)
    http_res = client.get("/v1/models/", cast_to=httpx.Response)
    assert http_res


def test_completions_create(llama_stack_client, default_model_id):
    client = OpenAIClientAdapter(llama_stack_client)
    response = client.completions.create(
        model=default_model_id,
        prompt="What is 2+2?",
        max_tokens=8,
        n=3,
        temperature=0.0,
    )
    assert response.choices
    assert len(response.choices) == 3


def test_chat_completions_create(llama_stack_client, default_model_id):
    client = OpenAIClientAdapter(llama_stack_client)
    response = client.chat.completions.create(
        model=default_model_id,
    )
    assert response
