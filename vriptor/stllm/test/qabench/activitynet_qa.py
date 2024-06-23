import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
from tqdm import tqdm

import argparse
import os
import torch
from stllm.common.config import Config
from stllm.common.registry import registry
from stllm.conversation.conversation import Chat, CONV_VIDEO_LLama2, CONV_VIDEO_Vicuna0, \
                    CONV_VISION_LLama2, CONV_instructblip_Vicuna0

# imports modules for registration
from stllm.datasets.builders import *
from stllm.models import *
from stllm.processors import *
from stllm.runners import *
from stllm.tasks import *
from stllm.test.video_utils import load_video_rawframes

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--frames", type=str, required=False, default=None)
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--ckpt-path", required=True, help="path to checkpoint file.")
    parser.add_argument("--num-frames", type=int, required=False, default=100)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    return parser.parse_args()


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    conv_dict = {'minigpt4_vicuna0': CONV_VIDEO_Vicuna0,
             "instructblip_vicuna0": CONV_instructblip_Vicuna0,
             "instructblip_vicuna0_btadapter": CONV_instructblip_Vicuna0,
             'minigpt4_vicuna0_btadapter': CONV_VIDEO_Vicuna0,}

    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_config.ckpt = args.ckpt_path
    model_cls = registry.get_model_class(model_config.arch)
    #model_config.eval = True
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    for name, para in model.named_parameters():
        para.requires_grad = False
    model.eval()
    
    CONV_VISION = conv_dict[model_config.model_type]
    model = model.to(torch.float16)

    chat = Chat(model, device='cuda:{}'.format(args.gpu_id))

    # Load both ground truth file containing questions and answers
    with open(args.gt_file_question) as file:
        gt_questions = json.load(file)
    with open(args.gt_file_answers) as file:
        gt_answers = json.load(file)
    
    if args.frames is not None:
        with open(args.frames,'r') as f:
            frames = json.load(f)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    
    for sample in tqdm(gt_questions):
        video_name = 'v_' + sample['video_name']
        question = sample['question']
        id = sample['question_id']
        answer = gt_answers[index]['answer']
        index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}
        video_path = os.path.join(args.video_dir, video_name)    
        # Check if the video exists
        chat_state = CONV_VISION.copy()
        img_list = []
        chat.upload_video(video_path, chat_state, img_list, args.num_frames, question)
        chat.ask(question, chat_state)
        llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=5,
                              do_sample=False,
                              temperature=1,
                              max_new_tokens=300,
                              max_length=2000)[0]

        sample_set['pred'] = llm_message
        output_list.append(sample_set)

    # Save the output list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
