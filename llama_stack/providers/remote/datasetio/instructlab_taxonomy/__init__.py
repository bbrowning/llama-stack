# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import TaxonomyDatasetIOConfig


async def get_adapter_impl(
    config: TaxonomyDatasetIOConfig,
    _deps,
):
    from .taxonomy import TaxonomyDatasetIOImpl

    impl = TaxonomyDatasetIOImpl(config)
    await impl.initialize()
    return impl
