import os
import copy
import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
import torch
from mmengine.fileio import FileClient

def load_video(vis_path, n_clips=1, num_frm=100):
    """
    Load video frames from a video file.

    Parameters:
    vis_path (str): Path to the video file.
    n_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """

    # Load video with VideoReader
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)

    # Currently, this function supports only 1 clip
    assert n_clips == 1

    # Calculate total number of frames to extract
    total_num_frm = min(total_frame_num, num_frm)
    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    # Extract frames as numpy array
    img_array = vr.get_batch(frame_idx)

    if isinstance(img_array, decord.ndarray.NDArray):
        img_array = img_array.asnumpy()
    else:
        img_array = img_array.numpy()
    
    img_array = img_array.reshape(
        (n_clips, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
    # Convert numpy arrays to PIL Image objects
    clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]

    return clip_imgs

def load_video_rawframes(vis_path, total_frame_num, n_clips=1, num_frm=100):
    # Currently, this function supports only 1 clip
    assert n_clips == 1
    # Calculate total number of frames to extract
    total_num_frm = min(total_frame_num, num_frm)
    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    # Extract frames as numpy array
    img_array = get_frames_from_raw(vis_path, frame_idx)
    # Set target image height and width
    target_h, target_w = 224, 224
    # If image shape is not as target, resize it
    if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

    # Reshape array to match number of clips and frames
    img_array = img_array.reshape(
        (n_clips, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
    # Convert numpy arrays to PIL Image objects
    clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]

    return clip_imgs

def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq

def get_frames_from_raw(directory, frame_idx, filename_tmpl="{:0>6}.jpg", offset=1):
    import mmcv
    mmcv.use_backend('cv2')
    file_client = FileClient('disk')
    imgs = list()
    cache = {}
    for i, frame_idx in enumerate(frame_idx):
        if frame_idx in cache:
            imgs.append(copy.deepcopy(imgs[cache[frame_idx]]))
            continue
        else:
            cache[frame_idx] = i
        frame_idx += offset
        filepath = os.path.join(directory, filename_tmpl.format(frame_idx))
        try:
            img_bytes = file_client.get(filepath)
        except:
            filepath = os.path.join(directory, filename_tmpl.format(frame_idx+1))
            img_bytes = file_client.get(filepath)
        cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        imgs.append(cur_frame)    
    return np.stack(imgs, axis=0)
