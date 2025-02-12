# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path
from typing import Any, Dict, List, Optional
import glob
import os
import tempfile

import datasets as hf_datasets
import git

from llama_stack.apis.datasetio import DatasetIO, PaginatedRowsResult
from llama_stack.apis.datasets import Dataset

from llama_stack.providers.datatypes import DatasetsProtocolPrivate
from llama_stack.providers.utils.datasetio.url_utils import get_dataframe_from_url
from llama_stack.providers.utils.kvstore import kvstore_impl

from instructlab.sdg.generate_data import preprocess_taxonomy
from instructlab.sdg.utils.json import jlload

from .config import TaxonomyDatasetIOConfig

DATASETS_PREFIX = "datasets:"


def load_taxonomy_dataset(dataset_def: Dataset, tempdir: str):
    local_path = os.path.join(tempdir, dataset_def.dataset_id)
    if dataset_def.metadata.get("path", None):
        clone_uri = dataset_def.metadata.get("path")
    else:
        clone_uri = dataset_def.url.uri
    # print(f"!!! cloning repo from {clone_uri} into {local_path}")
    git_repo = git.Repo.clone_from(clone_uri, local_path)

    qna_files = glob.glob(
        os.path.join("**", "qna.yaml"),
        root_dir=git_repo.working_dir,
        recursive=True,
    )
    # print(f"!!! found qna_files {qna_files}")
    rows = []
    for qna_file in qna_files:
        rows.append({
            "qna_path": qna_file,
            "qna_contents": Path(git_repo.working_dir).joinpath(qna_file).read_text(encoding="utf-8")
        })

    dataset = hf_datasets.Dataset.from_list(rows)
    # print(f"!!! made dataset {dataset}")

    # drop columns not specified by schema
    # if dataset_def.dataset_schema:
    #     dataset = dataset.select_columns(list(dataset_def.dataset_schema.keys()))

    return dataset


class TaxonomyDatasetIOImpl(DatasetIO, DatasetsProtocolPrivate):
    def __init__(self, config: TaxonomyDatasetIOConfig) -> None:
        self.config = config
        # local registry for keeping track of datasets within the provider
        self.dataset_infos = {}
        self.kvstore = None
        self.tempdir = tempfile.mkdtemp()

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.kvstore)
        # Load existing datasets from kvstore
        start_key = DATASETS_PREFIX
        end_key = f"{DATASETS_PREFIX}\xff"
        stored_datasets = await self.kvstore.range(start_key, end_key)

        for dataset in stored_datasets:
            dataset = Dataset.model_validate_json(dataset)
            self.dataset_infos[dataset.identifier] = dataset

    async def shutdown(self) -> None: ...

    async def register_dataset(
        self,
        dataset_def: Dataset,
    ) -> None:
        # Store in kvstore
        key = f"{DATASETS_PREFIX}{dataset_def.identifier}"
        await self.kvstore.set(
            key=key,
            value=dataset_def.json(),
        )
        self.dataset_infos[dataset_def.identifier] = dataset_def

    async def unregister_dataset(self, dataset_id: str) -> None:
        key = f"{DATASETS_PREFIX}{dataset_id}"
        await self.kvstore.delete(key=key)
        del self.dataset_infos[dataset_id]

    async def get_rows_paginated(
        self,
        dataset_id: str,
        rows_in_page: int,
        page_token: Optional[str] = None,
        filter_condition: Optional[str] = None,
    ) -> PaginatedRowsResult:
        dataset_def = self.dataset_infos[dataset_id]
        loaded_dataset = load_taxonomy_dataset(dataset_def, self.tempdir)

        if page_token and not page_token.isnumeric():
            raise ValueError("Invalid page_token")

        if page_token is None or len(page_token) == 0:
            next_page_token = 0
        else:
            next_page_token = int(page_token)

        start = next_page_token
        if rows_in_page == -1:
            end = len(loaded_dataset)
        else:
            end = min(start + rows_in_page, len(loaded_dataset))

        rows = [loaded_dataset[i] for i in range(start, end)]

        # print(f"!!! rows {rows}")
        return PaginatedRowsResult(
            rows=rows,
            total_count=len(rows),
            next_page_token=str(end),
        )

    async def append_rows(self, dataset_id: str, rows: List[Dict[str, Any]]) -> None:
        raise NotImplementedError("Appending to taxonomy datasets is not supported yet")
