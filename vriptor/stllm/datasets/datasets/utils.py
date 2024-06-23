#from utils.distributed import is_main_process, get_rank, get_world_size
import logging
import torch.distributed as dist
import torch
import io
import os
import json
import re
import copy
import numpy as np
from os.path import join
from tqdm import trange
from PIL import Image
from PIL import ImageFile
from torchvision.transforms import PILToTensor
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import random
import av
import cv2
import decord
import imageio
from decord import VideoReader
import torch
import math
decord.bridge.set_bridge("torch")

import logging
logger = logging.getLogger(__name__)

def load_image_from_path(image_path, client):
    if image_path.startswith('s3') or image_path.startswith('p2'):
        value = client.get(image_path)
        img_bytes = np.frombuffer(value, dtype=np.uint8)
        buff = io.BytesIO(img_bytes)
        image = Image.open(buff).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')  # PIL Image
    image = PILToTensor()(image).unsqueeze(0)  # (1, C, H, W), torch.uint8
    return image

def load_anno(ann_file_list):
    """[summary]

    Args:
        ann_file_list (List[List[str, str]] or List[str, str]):
            the latter will be automatically converted to the former.
            Each sublist contains [anno_path, image_root], (or [anno_path, video_root, 'video'])
            which specifies the data type, video or image

    Returns:
        List(dict): each dict is {
            image: str or List[str],  # image_path,
            caption: str or List[str]  # caption text string
        }
    """
    if isinstance(ann_file_list[0], str):
        ann_file_list = [ann_file_list]

    ann = []
    for d in ann_file_list:
        data_root = d[1]
        fp = d[0]
        is_video = len(d) == 3 and d[2] == "video"
        cur_ann = json.load(open(fp, "r"))
        iterator = trange(len(cur_ann), desc=f"Loading {fp}") \
            if is_main_process() else range(len(cur_ann))
        for idx in iterator:
            key = "video" if is_video else "image"
            # unified to have the same key for data path
            if isinstance(cur_ann[idx][key], str):
                cur_ann[idx]["image"] = join(data_root, cur_ann[idx][key])
            else:  # list
                cur_ann[idx]["image"] = [join(data_root, e) for e in cur_ann[idx][key]]
        ann += cur_ann
    return ann


def pre_text(text, max_l=None, pre_text=True):
    if pre_text:
        text = re.sub(r"([,.'!?\"()*#:;~])", '', text.lower())
        text = text.replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        text = re.sub(r"\s{2,}", ' ', text)
        text = text.rstrip('\n').strip(' ')

        if max_l:  # truncate
            words = text.split(' ')
            if len(words) > max_l:
                text = ' '.join(words[:max_l])
    else:
        pass
    return text


logger = logging.getLogger(__name__)


def collect_result(result, result_dir, filename, is_json=True, is_list=True):
    if is_json:
        result_file = os.path.join(
            result_dir, '%s_rank%d.json' % (filename, get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json' % filename)
        json.dump(result, open(result_file, 'w'))
    else:
        result_file = os.path.join(
            result_dir, '%s_rank%d.pth' % (filename, get_rank()))
        final_result_file = os.path.join(result_dir, '%s.pth' % filename)
        torch.save(result, result_file)

    dist.barrier()

    result = None
    if is_main_process():
        # combine results from all processes
        if is_list:
            result = []
        else:
            result = {}
        for rank in range(get_world_size()):
            if is_json:
                result_file = os.path.join(
                    result_dir, '%s_rank%d.json' % (filename, rank))
                res = json.load(open(result_file, 'r'))
            else:
                result_file = os.path.join(
                    result_dir, '%s_rank%d.pth' % (filename, rank))
                res = torch.load(result_file)
            if is_list:
                result += res
            else:
                result.update(res)

    return result


def sync_save_result(result, result_dir, filename, is_json=True, is_list=True):
    """gather results from multiple GPUs"""
    if is_json:
        result_file = os.path.join(
            result_dir, "dist_res", '%s_rank%d.json' % (filename, get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json' % filename)
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        json.dump(result, open(result_file, 'w'))
    else:
        result_file = os.path.join(
            result_dir, "dist_res", '%s_rank%d.pth' % (filename, get_rank()))
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        final_result_file = os.path.join(result_dir, '%s.pth' % filename)
        torch.save(result, result_file)

    dist.barrier()

    if is_main_process():
        # combine results from all processes
        if is_list:
            result = []
        else:
            result = {}
        for rank in range(get_world_size()):
            if is_json:
                result_file = os.path.join(
                    result_dir, "dist_res", '%s_rank%d.json' % (filename, rank))
                res = json.load(open(result_file, 'r'))
            else:
                result_file = os.path.join(
                    result_dir, "dist_res", '%s_rank%d.pth' % (filename, rank))
                res = torch.load(result_file)
            if is_list:
                result += res
            else:
                result.update(res)
        if is_json:
            json.dump(result, open(final_result_file, 'w'))
        else:
            torch.save(result, final_result_file)

        logger.info('result file saved to %s' % final_result_file)
    dist.barrier()
    return final_result_file, result


def pad_sequences_1d(sequences, dtype=torch.long, device=torch.device("cpu"), fixed_length=None):
    """ Pad a single-nested list or a sequence of n-d array (torch.tensor or np.ndarray)
    into a (n+1)-d array, only allow the first dim has variable lengths.
    Args:
        sequences: list(n-d tensor or list)
        dtype: np.dtype or torch.dtype
        device:
        fixed_length: pad all seq in sequences to fixed length. All seq should have a length <= fixed_length.
            return will be of shape [len(sequences), fixed_length, ...]
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=np.float32)
        >>> test_data_3d = [np.random.randn(2,3,4), np.random.randn(4,3,4), np.random.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=np.float32)
    """
    if isinstance(sequences[0], list):
        if "torch" in str(dtype):
            sequences = [torch.tensor(s, dtype=dtype, device=device) for s in sequences]
        else:
            sequences = [np.asarray(s, dtype=dtype) for s in sequences]

    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    if fixed_length is not None:
        max_length = fixed_length
    else:
        max_length = max(lengths)
    if isinstance(sequences[0], torch.Tensor):
        assert "torch" in str(dtype), "dtype and input type does not match"
        padded_seqs = torch.zeros((len(sequences), max_length) + extra_dims, dtype=dtype, device=device)
        mask = torch.zeros((len(sequences), max_length), dtype=torch.float32, device=device)
    else:  # np
        assert "numpy" in str(dtype), "dtype and input type does not match"
        padded_seqs = np.zeros((len(sequences), max_length) + extra_dims, dtype=dtype)
        mask = np.zeros((len(sequences), max_length), dtype=np.float32)

    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths

def pts_to_secs(pts: int, time_base: float, start_pts: int) -> float:
    """
    Converts a present time with the given time base and start_pts offset to seconds.

    Returns:
        time_in_seconds (float): The corresponding time in seconds.

    https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/utils.py#L54-L64
    """
    if pts == math.inf:
        return math.inf

    return int(pts - start_pts) * time_base


def get_pyav_video_duration(video_reader):
    video_stream = video_reader.streams.video[0]
    video_duration = pts_to_secs(
        video_stream.duration,
        video_stream.time_base,
        video_stream.start_time
    )
    return float(video_duration)


def get_frame_indices_by_fps():
    pass

def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices

def read_frames_av(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, clip=None,
    ):
    reader = av.open(video_path)
    frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
    vlen = len(frames)
    duration = get_pyav_video_duration(reader)
    fps = vlen / float(duration)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    frames = torch.stack([frames[idx] for idx in frame_indices])  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices, fps

def read_frames_gif(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, clip=None,
    ):
    if video_path.startswith('s3') or video_path.startswith('p2'):
        video_bytes = client.get(video_path)
        gif = imageio.get_reader(io.BytesIO(video_bytes))
    else:
        gif = imageio.get_reader(video_path)
    vlen = len(gif)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        max_num_frames=max_num_frames
    )
    frames = []
    for index, frame in enumerate(gif):
        # for index in frame_idxs:
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = torch.from_numpy(frame).byte()
            # # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
    frames = torch.stack(frames)  # .float() / 255
    return frames, frame_indices, 25. # for tgif

def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, clip=None
    ):
    if video_path.startswith('s3') or video_path.startswith('p2'):
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    if clip:
        frame_indices = [f + start_index for f in frame_indices]

    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices, float(duration)

def read_frames_decord_in_list(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, clip=None
    ):
    if not isinstance(video_path, list):
        video_path = [video_path]

    num_videos = len(video_path)
    if num_videos > 1:
        pass
    num_frames_per_video = num_frames // num_videos
    num_frames_per_video = [num_frames_per_video] * num_videos
    for i in range(num_frames % num_videos):
        num_frames_per_video[i] += 1

    all_frames = []
    all_frames_indices = []
    all_durations = 0
    previous_video_num_frames = 0

    for num_f, vpath in zip(num_frames_per_video, video_path):
        if vpath.startswith('s3') or vpath.startswith('p2'):
            video_bytes = client.get(vpath)
            video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
        else:
            video_reader = VideoReader(vpath, num_threads=1)
        vlen = len(video_reader)
        fps = video_reader.get_avg_fps()
        duration = vlen / float(fps)

        if clip:
            start, end = clip
            duration = end - start
            vlen = int(duration * fps)
            start_index = int(start * fps)

        frame_indices = get_frame_indices(
            num_f, vlen, sample=sample, fix_start=fix_start,
            input_fps=fps, max_num_frames=max_num_frames
        )
        if clip:
            frame_indices = [f + start_index for f in frame_indices]

        frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

        all_frames.append(frames)
        extend_frame_indices = [f + previous_video_num_frames for f in frame_indices]
        all_frames_indices.extend(extend_frame_indices)
        all_durations += duration
        previous_video_num_frames += vlen

    return torch.cat(all_frames, dim=0), all_frames_indices, all_durations

def read_frames_rawframes(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, clip=None
    ):
    file_client = client.FileClient('disk')
    fps = 5
    filename_tmpl="{:0>6}.jpg"
    offset=1
    frame_indices = get_frame_indices(
        num_frames, max_num_frames, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=-1
    )
    imgs = list()
    cache = {}
    for i, frame_idx in enumerate(frame_indices):
        # Avoid loading duplicated frames
        if frame_idx in cache:
            imgs.append(copy.deepcopy(imgs[cache[frame_idx]]))  
            continue
        else:
            cache[frame_idx] = i
        frame_idx += offset
        filepath = os.path.join(video_path, filename_tmpl.format(frame_idx))
        try:
            img_bytes = file_client.get(filepath)
        except:
            filepath = os.path.join(video_path, filename_tmpl.format(frame_idx+1))
            img_bytes = file_client.get(filepath)
        # Get frame with channel order RGB directly.
        import mmcv
        cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        imgs.append(cur_frame)
    frames = np.concatenate([img[np.newaxis, ...] for img in imgs], axis=0)
    frames = torch.from_numpy(frames)
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices, float(max_num_frames / fps)

VIDEO_READER_FUNCS = {
    'av': read_frames_av,
    'decord': read_frames_decord,
    'gif': read_frames_gif,
    'rawframe': read_frames_rawframes,
    'decord_list': read_frames_decord_in_list,
}
