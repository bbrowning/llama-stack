# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.inference import Inference
from llama_stack.apis.models import Models


class ServerLlamaStackClient:
    def __init__(self, inference_api: Inference, models_api: Models):
        self.inference = inference_api
        self.models = models_api
