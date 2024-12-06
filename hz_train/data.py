import os.path
from typing import List

import datasets
from torch.utils.data import Dataset

from .arguments import ModelDataarguments


class TrainDatasetForEmbedding(Dataset):
    def __init__(self, args: ModelDataarguments):
        if os.path.isdir(args.data_dir):
            # combine all the datasets in the directory
            train_datasets = []
            for file in os.listdir(args.data_dir):
                temp_dataset = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(args.data_dir, file),
                    split="train",
                    cache_dir=args.cache_dir_data,
                )

                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset(
                "json", data_files=args.data_dir, split="train"
            )

        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item: int) -> dict[str, str]:
        # Triplets, only support one positive and one negative example
        query = self.dataset[item]["query"]
        pos = self.dataset[item]["pos"][0]
        neg = self.dataset[item]["neg"][0]

        res = {"query": query, "pos": pos, "neg": neg}
        return res


class HzEmbeddingCollator:
    def __init__(self) -> None:
        pass

    def __call__(self, features: List[dict[str, str]]) -> dict[str, List[str]]:
        query = []
        pos = []
        neg = []

        for i in features:
            query.append(i["query"])
            pos.append(i["pos"])
            neg.append(i["neg"])

        res = {"query": query, "pos": pos, "neg": neg}
        return res
