import logging
import os
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from stllm.datasets.datasets.instruction_data import available_corpus, train_transform

import json
from os.path import basename
import numpy as np

from .utils import load_anno, pre_text, VIDEO_READER_FUNCS, load_image_from_path

try:
    from mmengine import fileio 
    has_client = True
except ImportError:
    has_client = False

logger = logging.getLogger(__name__)


class ImageVideoBaseDataset(Dataset):
    """Base class that implements the image and video loading methods"""

    media_type = "video"

    def __init__(self):
        assert self.media_type in ["image", "video", "only_video"]
        self.data_root = None
        self.anno_list = (
            None  # list(dict), each dict contains {"image": str, # image or video path}
        )
        self.transform = None
        self.video_reader = None
        self.num_tries = None

        self.client = None
        if has_client:
            self.client = fileio

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_anno(self, index):
        """obtain the annotation for one media (video or image)

        Args:
            index (int): The media index.

        Returns: dict.
            - "image": the filename, video also use "image".
            - "caption": The caption for this file.

        """
        anno = self.anno_list[index]
        if self.data_root is not None:
            anno["image"] = os.path.join(self.data_root, anno["image"])
        return anno

    def load_and_transform_media_data(self, index, data_path):
        if self.media_type == "image":
            return self.load_and_transform_media_data_image(index, data_path)
        else:
            return self.load_and_transform_media_data_video(index, data_path)

    def load_and_transform_media_data_image(self, index, data_path):
        image = load_image_from_path(data_path, client=self.client)
        image = self.transform(image)
        return image, index

    def load_and_transform_media_data_video(self, index, data_path, return_fps=False, clip=None):
        for _ in range(self.num_tries):
            try:
                max_num_frames = self.max_num_frames if hasattr(self, "max_num_frames") else -1
                frames, frame_indices, sec = self.video_reader(
                    data_path, self.num_frames, self.sample_type, 
                    max_num_frames=max_num_frames, client=self.client, clip=clip
                )
            except Exception as e:
                logger.warning(
                    f"Caught exception {e} when loading video {data_path}, "
                    f"randomly sample a new video as replacement"
                )
                index = random.randint(0, len(self) - 1)
                ann = self.get_anno(index)
                data_path = ann["image"]
                continue
            # shared aug for video frames
            frames = self.transform(frames)
            if return_fps:
                #sec = [str(round(f / fps, 1)) for f in frame_indices]
                return frames, index, sec
            else:
                return frames, index
        else:
            raise RuntimeError(
                f"Failed to fetch video after {self.num_tries} tries. "
                f"This might indicate that you have many corrupted videos."
            )

class PTImgTrainDataset(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(self, ann_file, transform, pre_text=True):
        super().__init__()

        if len(ann_file) == 3 and ann_file[2] == "video":
            self.media_type = "video"  
        else:
            self.media_type = "image"
        self.label_file, self.data_root = ann_file[:2]

        logger.info('Load json file')
        with open(self.label_file, 'r') as f:
            self.anno = json.load(f)
        self.num_examples = len(self.anno)

        self.transform = transform
        self.pre_text = pre_text
        logger.info(f"Pre-process text: {pre_text}")

    def get_anno(self, index):
        filename = self.anno[index][self.media_type]
        caption = self.anno[index]["caption"]
        anno = {"image": os.path.join(self.data_root, filename), "caption": caption}
        return anno

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            image, index = self.load_and_transform_media_data(index, ann["image"])
            caption = pre_text(ann["caption"], pre_text=self.pre_text)
            return image, caption, index
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading image {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)

class PTVidTrainDataset(PTImgTrainDataset):
    media_type = "video"

    def __init__(
        self,
        ann_file,
        transform,
        num_frames=4,
        video_reader_type="decord",
        sample_type="rand",
        num_tries=3,
        pre_text=True
    ):
        super().__init__(ann_file, transform, pre_text=pre_text)
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries

class ITImgTrainDataset(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(
        self, ann_file, transform, simple=False,
        system="", role=("Human", "Assistant"),
        start_token="<Image>", end_token="</Image>",
        random_shuffle=True, # if True, shuffle the QA list
    ):
        super().__init__()

        if len(ann_file) == 3 and ann_file[2] == "video":
            self.media_type = "video"  
        else:
            self.media_type = "image"
        self.label_file, self.data_root = ann_file[:2]

        logger.info('Load json file')
        with open(self.label_file, 'r') as f:
            self.anno = json.load(f)
        self.num_examples = len(self.anno)
        self.transform = transform

        # prompt parameters
        if system:
            assert system[-1] == " ", "' ' should be add in the end of system, thus '###' will be tokenized into one token."
        # currently not support add start_token and end_token in the system, since the msg should be added properly
        self.begin_signal = "###"
        self.end_signal = " "
        self.start_token = start_token
        self.end_token = end_token
        self.system = system
        self.role = role
        self.random_shuffle = random_shuffle
        self.simple = simple
        # instruction location and number
        logger.info(f"Random shuffle: {self.random_shuffle}")

    def get_anno(self, index):
        filename = self.anno[index][self.media_type]
        qa = self.anno[index]["QA"]
        if "num_frames" in self.anno[index]:
            self.max_num_frames = self.anno[index]["num_frames"]
        if "start" in self.anno[index] and "end" in self.anno[index]:
            anno = {
                "image": os.path.join(self.data_root, filename), "qa": qa,
                "start": self.anno[index]["start"], "end": self.anno[index]["end"],
            }
        else:
            anno = {"image": os.path.join(self.data_root, filename), "qa": qa}
        return anno

    def __len__(self):
        return self.num_examples
    
    def process_qa(self, qa, msg=""):
        cur_instruction = ""
        # randomly shuffle qa for conversation
        if self.random_shuffle and len(qa) > 1:
            random.shuffle(qa)
        if "i" in qa[0].keys() and qa[0]["i"] != "":
            cur_instruction = qa[0]["i"] + self.end_signal

        conversation = self.system
        # add instruction as system message

        # rstrip() for the extra " " in msg
        if not self.simple:
            if cur_instruction:
                conversation += cur_instruction
            conversation += (
                self.begin_signal + self.role[0] + ": " + 
                self.start_token + '<ImageHere>' + self.end_token + msg.rstrip() + ' ' + 
                qa[0]["q"] + self.end_signal + self.begin_signal + self.role[1] + ": "
            )
        else:
            conversation += '<ImageHere>'
            conversation += (
                self.begin_signal + self.role[0] + ": " + cur_instruction + msg.rstrip() + 
                qa[0]["q"] + self.end_signal + self.begin_signal + self.role[1] + ": "
            )
        
        return conversation, qa[0]["a"]

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            image, index = self.load_and_transform_media_data_image(index, ann["image"])
            instruction, answer = self.process_qa(ann["qa"])
            return {
                "image": image,
                "answer": answer,
                "image_id": index,
                "instruction_input": instruction
            }
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading image {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)

class ITVidTrainDataset(ITImgTrainDataset):
    media_type = "video"

    def __init__(
        self, ann_file, transform, simple=False,
        num_frames=4, video_reader_type="decord", sample_type="rand", num_tries=3,
        system="", role=("Human", "Assistant"),
        start_token="<Video>", end_token="</Video>",
        add_second_msg=False,
        random_shuffle=True,
    ):
        super().__init__(
            ann_file, transform, 
            system=system, role=role,
            start_token=start_token, end_token=end_token,
            random_shuffle=random_shuffle,
            simple=simple,
        )
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.add_second_msg = add_second_msg

        logger.info(f"Use {video_reader_type} for data in {ann_file}")
        if add_second_msg:
            logger.info(f"Add second message: The video contains X frames sampled at T seconds.")

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            msg = ""
            clip = None
            if "start" in ann and "end" in ann:
                clip = [ann["start"], ann["end"]]
            video, index, sec = self.load_and_transform_media_data_video(index, ann["image"], return_fps=True, clip=clip)
            if self.add_second_msg:
                # " " should be added in the start and end
                msg = f" The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds. "
            instruction, answer = self.process_qa(ann["qa"], msg)
            return {
                "image": video,
                "answer": answer,
                "image_id": index,
                "instruction_input": instruction,
                "video_len": sec
            }
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading video {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)
 
if __name__ == "__main__":
    pass

