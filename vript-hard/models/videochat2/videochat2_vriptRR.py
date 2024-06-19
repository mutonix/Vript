from utils.config import Config
config_file = "configs/config.json"
cfg = Config.from_file(config_file)

import os
import io
import json
import csv

from models.videochat2_it import VideoChat2_it
from utils.easydict import EasyDict
import torch

from transformers import StoppingCriteria, StoppingCriteriaList

from PIL import Image
import numpy as np
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset

from torchvision import transforms

import matplotlib.pyplot as plt

from tqdm import tqdm

from IPython.display import Video, HTML

from peft import get_peft_model, LoraConfig, TaskType
import copy


from argparse import ArgumentParser
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path to the model checkpoint", default='your_path_to_model/videochat2_7b_stage3.pth')
    parser.add_argument("--rr_annotation_file", type=str, help="path to the RR annotation file", default='your_path_to_dataset/RR_annotations.jsonl')
    parser.add_argument("--rr_video_path", type=str, help="path to the RR video file", default='your_path_to_dataset/RR_videos')
    parser.add_argument("--rr_clip_path", type=str, help="path to the RR clip file", default='your_path_to_dataset/RR_scenes')
    parser.add_argument("--output_filename_video", type=str, help="output filename for video", default='your_path_to_output/RR_video_output.csv')
    parser.add_argument("--output_filename_clip", type=str, help="output filename for clip", default='your_path_to_output/RR_clip_output.csv')
    args = parser.parse_args()
    return args

args = parse_args()

# load stage2 model
cfg.model.vision_encoder.num_frames = 4
model = VideoChat2_it(config=cfg.model)

model = model.to(torch.device(cfg.device))
model = model.eval()

# add lora to run stage3 model
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, 
    r=16, lora_alpha=32, lora_dropout=0.
)
model.llama_model = get_peft_model(model.llama_model, peft_config)

state_dict = torch.load(args.model_path, "cpu")

if 'model' in state_dict.keys():
    msg = model.load_state_dict(state_dict['model'], strict=False)
else:
    msg = model.load_state_dict(state_dict, strict=False)
print(msg)

model = model.eval()

def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + ": " + message + conv.sep
        else:
            ret += role + ":"
    return ret


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + ": " + message
        else:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
    return ret

def get_context_emb(conv, model, img_list, answer_prompt=None, print_res=False):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    if print_res:
        print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    with torch.no_grad():
        seg_tokens = [
            model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.llama_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs


def ask(text, conv):
    conv.messages.append([conv.roles[0], text + '\n'])
        

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([835]).to("cuda:0"),
        torch.tensor([2277, 29937]).to("cuda:0")]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        outputs = model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    conv.messages[-1][1] = output_text
    return output_text, output_token.cpu().numpy()

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    # transform
    crop_size = resolution
    scale_size = resolution
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return torch_imgs, msg
    else:
        return torch_imgs
    
def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    print(f"n_position: {n_position}")
    print(f"pre_n_position: {pre_n_position}")
    
    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        if new_P != 14:
            print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if cur_frame != ckpt_num_frame:

        print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        print(f'Interpolate the position embedding')
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table

# infer_vriptRR
def infer_vriptRR(
        data_sample, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        hint_prompt='',  # add in the end of question prompt
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False
    ):
    # video preprocess
    num_frame = 16
    resolution = 224
    vid, msg = load_video(data_sample['video_path'], num_segments=num_frame, return_msg=True, resolution=resolution)
    new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frame, cur_frame=num_frame)
    model.vision_encoder.encoder.pos_embed = new_pos_emb
    TC, H, W = vid.shape
    video = vid.reshape(1, TC//3, 3, H, W).to("cuda:0")
  
    video_list = []
    with torch.no_grad():
        if system_q:
            video_emb, _ = model.encode_img(video, system + data_sample['question'])
        else:
            video_emb, _ = model.encode_img(video, system)
    video_list.append(video_emb)


    chat = EasyDict({
        "system": system,
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })

    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video>\n"])
    
    if system_llm:
        prompt = system + data_sample['question'] + hint_prompt + data_sample['hint'] + question_prompt
    else:
        prompt = data_sample['question'] + hint_prompt + data_sample['hint'] + question_prompt
    
    ask(prompt, chat)

    llm_message = answer(
        conv=chat, model=model, do_sample=False, 
        img_list=video_list, max_new_tokens=100, 
        answer_prompt=answer_prompt, print_res=print_res
    )[0]
    # remove potential explanation
    llm_message = return_prompt + llm_message.strip().split('\n')[0]
    print(llm_message)
    print(f"GT: {data_sample['answer']}")
    return llm_message

def check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
        
    return flag

system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
question_prompt="\nOnly give the best option."
answer_prompt="Best option:("
return_prompt='('
hint_prompt="\nHint:"

vript_RR_annotation_file = args.rr_annotation_file
vript_RR_video_path = args.rr_video_path
vript_RR_video_clip_path = args.clip_path
output_filename_video = args.output_filename_video
output_filename_clip = args.output_filename_clip


with open(vript_RR_annotation_file, 'r') as f:
    vript_RR_dataset = [json.loads(line) for line in f]

correct_video = 0
correct_clip = 0
total = 0
results = []


results_video = []
results_clip = []
for data_sample in tqdm(vript_RR_dataset):
    video_id = data_sample['video_id']
    video_path = os.path.join(vript_RR_video_path, video_id + '.mp4')

    clip_id = data_sample['clip_id']    
    clip_path = os.path.join(vript_RR_video_clip_path, clip_id + '.mp4')

    question = data_sample['question']
    hint = data_sample['hint']
    muliple_choice = data_sample['multiple_choice']
    muliple_choice_answer = data_sample['multiple_choice_answer']


    # 多选数据格式处理
    question = f"Question: {question}\n"
    question += "Options:\n"
    answer_idx = -1
    for idx, c in enumerate(muliple_choice):
        question += f"({chr(ord('A') + idx)}) {c}\n"
        if c == muliple_choice_answer:
            answer_idx = idx
    question = question.rstrip()
    muliple_choice_answer = f"({chr(ord('A') + answer_idx)}) {muliple_choice_answer}"



    data_item_video = {'video_path': video_path, 'question': question, 'answer': muliple_choice_answer, 'hint': hint}


    data_item_clip = {'video_path': clip_path, 'question': question, 'answer': muliple_choice_answer, 'hint': hint}

    

    pred_video = infer_vriptRR(data_item_video, system=system, question_prompt=question_prompt, answer_prompt=answer_prompt, return_prompt=return_prompt, hint_prompt=hint_prompt, system_q=False, print_res=True, system_llm=True)
    pred_clip = infer_vriptRR(data_item_clip, system=system, question_prompt=question_prompt, answer_prompt=answer_prompt, return_prompt=return_prompt, hint_prompt=hint_prompt, system_q=False, print_res=True, system_llm=True)

    gt = data_item_video['answer']
    print(pred_video)
    print(pred_clip)
    print(gt)
    results_video.append([data_sample["video_id"], pred_video, gt, data_sample])
    results_clip.append([data_sample["clip_id"], pred_clip, gt, data_sample])
    if check_ans(pred=pred_clip, gt=gt):
        correct_clip += 1
    if check_ans(pred=pred_video, gt=gt):
        correct_video += 1
    total += 1

print(f'Correct clip: {correct_clip}/{total}')
print(f'Correct video: {correct_video}/{total}')


import csv
with open(output_filename_video, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'pred', 'gt', 'meta'])
    for item in results_video:
        writer.writerow(item)
with open(output_filename_clip, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'pred', 'gt', 'meta'])
    for item in results_clip:
        writer.writerow(item)

