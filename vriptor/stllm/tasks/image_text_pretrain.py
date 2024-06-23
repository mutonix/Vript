"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from stllm.common.registry import registry
from stllm.tasks.base_task import BaseTask
from stllm.datasets.datasets.instruction_data import available_corpus, train_transform
from stllm.datasets.datasets.image_video_itdatasets import ITImgTrainDataset, ITVidTrainDataset

@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass

@registry.register_task("video_text_it")
class VideoTextItTask(ImageTextPretrainTask):
    def __init__(self):
        super().__init__()

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """
        datasets = dict()
        datasets_config = cfg.datasets_cfg
        assert len(datasets_config) > 0, "At least one dataset has to be specified."
        simple = cfg.model_cfg.get('qformer_text_input',False)
        for name in datasets_config:
            dataset_config = datasets_config[name]
            dataset_info = available_corpus[name]
            dataset_cls = ITImgTrainDataset if get_media_type(dataset_info)=="image" else ITVidTrainDataset

            datasets[name] = {'train': dataset_cls(ann_file=dataset_info, simple=simple,
                        transform=train_transform, **dataset_config)}

        return datasets

def get_media_type(dataset_info):
    if len(dataset_info) == 3 and dataset_info[2] == "video":
        return "video"
    else:
        return "image"
