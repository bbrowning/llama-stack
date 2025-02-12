# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.providers.datatypes import Api, InlineProviderSpec, ProviderSpec


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.synthetic_data_generation,
            provider_type="inline::instructlab-sdg",
            pip_packages=["instructlab-sdg>=0.7.0"],
            module="llama_stack.providers.inline.synthetic_data_generation.instructlab_sdg",
            config_class="llama_stack.providers.inline.synthetic_data_generation.instructlab_sdg.synthetic_data_generation.InstructLabSDGConfig",
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
                Api.inference,
            ],
        ),
    ]
