# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .distributed_iterable_dataset import DistributedIterableDataset


class InterleavedBaseIterableDataset(DistributedIterableDataset):

    def _init_data(self):
        data = {
            'sequence_plan': [],
            'text_ids_list': [],
            'image_tensor_list': [],
            'num_tokens': 0,
        }
        return data

    def _add_text(self, data, text, need_loss, enable_cfg=True):
        text_ids = self.tokenizer.encode(text)
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append(
            {
                'type': 'text',
                'enable_cfg': int(enable_cfg),
                'loss': int(need_loss),
                'special_token_loss': 0,
                'special_token_label': None,
            }
        )
        return data

    def _add_image(self, data, image, need_loss, need_vae, need_vit, enable_cfg=True,
                   target_transform=None):
        assert need_loss or need_vae or need_vit

        if need_loss:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image', 
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )

            transform = self.transform if target_transform is None else target_transform
            image_tensor = transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // transform.stride ** 2
            data['image_tensor_list'].append(image_tensor)

        if need_vae:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image', 
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor.clone())

        if need_vit:
            data['sequence_plan'].append(
                {
                    'type': 'vit_image',
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
            )
            vit_image_tensor = self.vit_transform(image)
            height, width = vit_image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.vit_transform.stride ** 2
            data['image_tensor_list'].append(vit_image_tensor)

        return data
