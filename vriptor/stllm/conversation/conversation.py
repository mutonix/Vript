import argparse
import time
import numpy as np
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

from stllm.common.registry import registry
from stllm.test.video_utils import load_video
import torchvision.transforms as T
from stllm.test.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    instruction: bool
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            instruction=self.instruction,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

def get_residual_index(sample_segments, total_segments, devices):
    seg_size = float(total_segments) / sample_segments
    frame_indices = np.array([
    int((seg_size / 2) + np.round(seg_size * idx))
    for idx in range(sample_segments)
    ])
    frame_indices = torch.from_numpy(frame_indices).to(devices)
    return frame_indices

CONV_VISION_Vicuna0 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human: ", "Assistant: "),
    messages=[],
    offset=2,
    instruction=True,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

CONV_VIDEO_Vicuna0 = Conversation(
    system="Give the following video: <Video>VideoContent</Video>. "
           "You will be able to see the video once I provide it to you. Please answer my questions.",
    roles=("Human: ", "Assistant: "),
    messages=[],
    offset=2,
    instruction=True,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

CONV_instructblip_Vicuna0 = Conversation(
    system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, give your answer that best addresses the question.\n",
    roles=("Human: ", "Assistant: "),
    messages=[],
    instruction=False,
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

CONV_VISION_LLama2 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("<s>[INST] ", " [/INST] "),
    messages=[],
    offset=2,
    instruction=True,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

CONV_VIDEO_LLama2 = Conversation(
    system="Give the following video: <Img>VideoContent</Img>. "
           "You will be able to see the video once I provide it to you. Please answer my questions.",
    roles=("<s>[INST] ", " [/INST] "),
    messages=[],
    offset=2,
    instruction=True,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

class Chat:
    def __init__(self, model, device='cuda:0'):
        self.device = device
        self.model = model
        if not hasattr(model,'llama_model'):
            if hasattr(model.model,'stllm_model'):
                self.model = model.model.stllm_model
            else:
                self.model = model.model.model.stllm_model
            self.LLM = model

        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(224), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(224),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and (conv.messages[-1][1][-6:] == '</Img>' or conv.messages[-1][1][-8:] == '</Video>' 
                    or conv.messages[-1][1][-8:] == '</Frame>'):  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9, system=True,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000, do_sample=True):
        conv.append_message(conv.roles[1], None)
        if conv.instruction:
            embs, attention_mask = self.get_context_emb(conv, img_list)
        else:
            embs, attention_mask = self.get_context_emb_sim(conv, img_list, system=system)
            repetition_penalty = 1.5

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        llama_model = self.LLM if hasattr(self,'LLM') else self.model.llama_model
        outputs = llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            #attention_mask=attention_mask,
            stopping_criteria=self.stopping_criteria,
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
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def upload_img(self, image, conv, img_list):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.transform([raw_image]).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.transform([raw_image]).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        image_emb, _ = self.model.encode_img(image)
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg

    def upload_video(self, video, conv, img_list, num_frame=64, text=None):
        raw_frames = load_video(video, num_frm=num_frame) if isinstance(video,str) else video
        video_frames = self.transform(raw_frames).to(self.device) 
        bt, w, h = video_frames.size()
        video_frames = video_frames.view(bt//3,3,w,h)

        video_emb, _, _ = self.model.encode_img(video_frames, text=text)
        if self.model.video_input == 'mean':
            video_emb = video_emb.mean(dim=0, keepdim=True)
        elif self.model.video_input == 'all':
            video_emb = video_emb.view(1, -1, video_emb.size(-1))
        elif self.model.video_input == 'residual':
            T = video_emb.size(0)
            residual_size = self.model.residual_size
            residual_index = get_residual_index(residual_size, T, video_emb.device)
            global_embeds = video_emb.mean(dim=0, keepdim=True)
            local_embeds = video_emb[residual_index]
            global_embeds = global_embeds.expand((residual_size,-1,-1)).to(self.model.up_proj.weight.dtype)
            global_embeds = self.model.up_proj(self.model.non_linear_func(self.model.down_proj(global_embeds)))
            video_emb = (local_embeds + global_embeds).view(1,-1,video_emb.size(-1)).contiguous()
        
        img_list.append(video_emb)
        sign='<Video><ImageHere></Video>'
        conv.append_message(conv.roles[0], sign)
        msg = "Received."
        return msg
    
    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        if hasattr(self.model, "embed_tokens"):
            embed_tokens = self.model.embed_tokens
        elif hasattr(self.model.llama_model.model, "embed_tokens"):
            embed_tokens = self.model.llama_model.model.embed_tokens
        else:
            embed_tokens = self.model.llama_model.model.model.embed_tokens
        seg_embs = [embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs, None
    
    def get_context_emb_sim(self, conv, img_list, system=True):
        question = conv.messages[0][1]
        question = question.split('</Video> ')[1]
        system = conv.system if system else ""
        # question = system + "###Human: " + question + " ###Assistant: "
        seg_tokens = self.model.llama_tokenizer(
                [question], return_tensors="pt", add_special_tokens=0 == 0).to(self.device)
        
        if hasattr(self.model, "embed_tokens"):
            embed_tokens = self.model.embed_tokens
        elif hasattr(self.model.llama_model.model, "embed_tokens"):
            embed_tokens = self.model.llama_model.model.embed_tokens
        else:
            embed_tokens = self.model.llama_model.model.model.embed_tokens
        seg_embs = embed_tokens(seg_tokens.input_ids) 
        mixed_embs = torch.cat((img_list[0],seg_embs), dim=1)
        atts_img = torch.ones(img_list[0].size()[:-1], dtype=torch.long).to(mixed_embs.device)
        attention_mask = torch.cat([atts_img, seg_tokens.attention_mask], dim=1)
        return mixed_embs, attention_mask
        


