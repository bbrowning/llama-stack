# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional
import asyncio
import inspect

from llama_stack.apis.common.content_types import InterleavedContent
from llama_stack.apis.inference import Inference, LogProbConfig, ResponseFormat, StopReason
from llama_stack.apis.models import Models
from llama_stack.models.llama.datatypes import SamplingParams


def _ensure_sync(result):
    if inspect.iscoroutine(result):
        async def async_result():
            return await result
        return asyncio.run(async_result())
    return result


class SyncInference:
    def __init__(self, inference_api: Inference):
        self.inference_api = inference_api


    def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ):
        result = self.inference_api.completion(
            model_id=model_id,
            content=content,
            sampling_params=sampling_params,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )
        result = _ensure_sync(result)

        # Convert StopReason enum to string values
        if isinstance(result.stop_reason, StopReason):
            result.stop_reason = result.stop_reason.value

        return result


class SyncModels:
    def __init__(self, models_api: Models):
        self.models_api = models_api


class SyncServerLlamaStackClient:
    def __init__(self, inference_api: Inference, models_api: Models):
        self.inference = SyncInference(inference_api)
        self.models = SyncModels(models_api)
