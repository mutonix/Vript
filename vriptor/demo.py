import argparse
import torch
import os
import subprocess

from stllm.common.config import Config
from stllm.common.registry import registry
from stllm.conversation.conversation import Chat, CONV_instructblip_Vicuna0

# imports modules for registration
from stllm.datasets.builders import *
from stllm.models import *
from stllm.processors import *
from stllm.runners import *
from stllm.tasks import *
import decord


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='config/vriptor_stllm_stage2.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--ckpt-path", default='model_weights/vriptor_stllm_stage2', help="path to the model checkpoint.")
    parser.add_argument("--video-path", default='video_examples/emoji.mp4', help="path to the video.")
    parser.add_argument("--voice-enable", action='store_true', help="enable voiceover")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

ckpt_path = args.ckpt_path
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_config.ckpt = ckpt_path
model_config.llama_model = ckpt_path
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.to(torch.float16)
CONV_VISION = CONV_instructblip_Vicuna0

if args.voice_enable:
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel('small', device="cuda", device_index=args.gpu_id, compute_type="int8", num_workers=16, download_root='whisper')

def process_audio(out_folder, video):
    os.makedirs(out_folder, exist_ok=True)
    audio_file = os.path.join(out_folder, f'{video}.mp3')
    print(f"Processing audio from {video}...")
    out = subprocess.call(['ffmpeg', '-y', '-i', video, '-vn', '-c:a', 'mp3', audio_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if out != 0:
        return None
    segments, info = whisper_model.transcribe(audio_file, beam_size=3, vad_filter=True)
    
    voiceover = ""
    for seg in segments:
        span = [seg.start, seg.end]
        text = seg.text
        voiceover += f"[{round(float(span[0]), 1)}, {round(float(span[1]), 1)}] " + text.strip() + "\n"
  
    if os.path.exists(audio_file):
        os.remove(audio_file)
  
    return voiceover

chat = Chat(model, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

chat_state = CONV_VISION.copy()
video = args.video_path
video_reader = decord.VideoReader(video)
video_fps = video_reader.get_avg_fps()
video_frames = len(video_reader)    
duration = video_frames / video_fps

if args.voice_enable:
    voiceover = process_audio('audio', video)
else:
    voiceover = None

user_message = "scene by scene" # "scene by scene" or "whole video"

if voiceover:
    if 'scene by scene' in user_message.lower():
        user_message = f'Voiceover:"{voiceover}". Based on the video and voiceover, describe the video scene by scene in detail (Duration: [0.0, {duration:.1f}]s, Sampling Rate: {duration / 64:.2f}s/frame).'
    else:
        user_message = f'Voiceover:"{voiceover}". Based on the video and voiceover, describe the video in detail (Duration: [0.0, {duration:.1f}]s, Sample Rate: {duration / 64:.2f}s/frame).'
else:
    if 'scene by scene' in user_message.lower():
        user_message = f'Describe the video scene by scene in detail (Duration: [0.0, {duration:.1f}]s, Sample Rate: {duration / 64:.2f}s/frame).'
    else:
        user_message = f'Describe the video in detail (Duration: [0.0, {duration:.1f}]s, Sample Rate: {duration / 64:.2f}s/frame).'

prompt = user_message
print(f"\n{prompt}\n")
img_list = []

chat.upload_video(video, chat_state, img_list, 64, text=prompt)
chat.ask("###Human: " + prompt + " ###Assistant: ", chat_state)
llm_message = chat.answer(conv=chat_state,
                img_list=img_list,
                num_beams=3,
                do_sample=True,
                max_new_tokens=2048,
                top_p=0.9,
                repetition_penalty=1.5,
                max_length=8192)[0]
print(llm_message)


