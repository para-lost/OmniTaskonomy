"""Release manifests adapted to BAGEL's original image/text packing helpers."""

from copy import deepcopy
from pathlib import Path
import random

from PIL import Image
import torch
from torchvision.transforms import InterpolationMode

from data.interleaved_base import InterleavedBaseIterableDataset
from omnitaskonomy.data.common import read_jsonl, resolve_image


class ManifestDataset(InterleavedBaseIterableDataset):
    def __init__(self, dataset_name, manifest, kind, transform, tokenizer,
                 vit_transform, local_rank=0, world_size=1, num_workers=1,
                 data_status=None, num_used_data=None, data_seed=None,
                 shuffle_before_slice=False, target_interpolation="bicubic"):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        if kind not in {"i2i", "i2t"}:
            raise ValueError(f"Unknown manifest kind: {kind}")
        if target_interpolation not in {"bicubic", "nearest"}:
            raise ValueError(f"Unknown target interpolation: {target_interpolation}")
        if data_status is not None:
            raise ValueError("Release runs restart from model weights, not partial dataloader state")
        self.manifest = Path(manifest).resolve()
        self.kind = kind
        self.transform = transform
        self.target_transform = transform
        if target_interpolation == "nearest":
            self.target_transform = deepcopy(transform)
            self.target_transform.resize_transform.interpolation = InterpolationMode.NEAREST
            self.target_transform.resize_transform.antialias = False
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_seed = data_seed
        self.shuffle_before_slice = shuffle_before_slice
        self.num_used_data = num_used_data
        self.rows = list(read_jsonl(self.manifest))
        if num_used_data is not None:
            if num_used_data > len(self.rows):
                raise ValueError(f"{self.manifest}: requested pool {num_used_data}, found {len(self.rows)} rows")
            if not shuffle_before_slice:
                self.rows = self.rows[:num_used_data]
        pool_size = len(self.rows) if num_used_data is None else num_used_data
        if pool_size < world_size * max(1, num_workers):
            raise ValueError("Manifest needs at least one row per distributed data worker")
        uids = [row["uid"] for row in self.rows]
        if len(set(uids)) != len(uids):
            raise ValueError(f"Duplicate uid in {self.manifest}")
        self.set_epoch()

    def set_epoch(self, seed=42):
        self.indices = list(range(len(self.rows)))
        random.Random(seed if self.data_seed is None else self.data_seed).shuffle(self.indices)
        if self.shuffle_before_slice and self.num_used_data is not None:
            self.indices = self.indices[:self.num_used_data]

    def _image(self, value):
        with Image.open(resolve_image(self.manifest, value)) as image:
            return image.convert("RGB")

    def parse_row(self, row):
        data = self._init_data()
        if self.kind == "i2i":
            self._add_image(data, self._image(row["source_image"]),
                            need_loss=False, need_vae=True, need_vit=True)
            self._add_text(data, row["prompt"], need_loss=False)
            self._add_image(data, self._image(row["target_image"]),
                            need_loss=True, need_vae=False, need_vit=False, enable_cfg=False,
                            target_transform=self.target_transform)
            return data

        conversations = row["conversations"]
        if not conversations or conversations[0]["from"] != "human":
            raise ValueError(f"{row['uid']}: conversations must start with a human turn")
        placeholders = sum(turn["value"].count("<image>") for turn in conversations)
        if "images" in row:
            image_values = row["images"]
            if not image_values or placeholders != len(image_values):
                raise ValueError(f"{row['uid']}: images must match <image> placeholders")
        else:
            image_values = [row["image"]]
            if placeholders > 1:
                raise ValueError(f"{row['uid']}: single-image manifests accept at most one <image> token")
        images = iter(self._image(value) for value in image_values)
        if not placeholders:
            self._add_image(data, next(images), need_loss=False, need_vae=False,
                            need_vit=True, enable_cfg=False)
        has_answer = False
        for turn in conversations:
            role, value = turn["from"], turn["value"]
            if role == "gpt":
                if "<image>" in value or not value.strip():
                    raise ValueError(f"{row['uid']}: invalid assistant answer")
                self._add_text(data, value, need_loss=True, enable_cfg=False)
                has_answer = True
            elif role == "human":
                parts = value.split("<image>")
                for index, part in enumerate(parts):
                    text = part.strip() if len(parts) > 1 else part
                    if text:
                        self._add_text(data, text, need_loss=False, enable_cfg=False)
                    if index < len(parts) - 1:
                        self._add_image(data, next(images), need_loss=False, need_vae=False,
                                        need_vit=True, enable_cfg=False)
            else:
                raise ValueError(f"{row['uid']}: unsupported conversation role {role!r}")
        if not has_answer:
            raise ValueError(f"{row['uid']}: no supervised assistant answer")
        return data

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        workers, worker_id = (info.num_workers, info.id) if info else (1, 0)
        worker_rank = self.local_rank * workers + worker_id
        indices = self.indices[worker_rank::self.world_size * workers]
        if not indices:
            raise ValueError("Distributed worker has no manifest rows")
        while True:
            for index in indices:
                row = self.rows[index]
                data = self.parse_row(row)
                data["data_indexes"] = {"data_indexes": index, "worker_id": worker_id,
                                        "dataset_name": self.dataset_name, "uid": row["uid"]}
                yield data
