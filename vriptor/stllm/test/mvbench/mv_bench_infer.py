import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
from tqdm import tqdm
from mv_bench import MVBench_dataset, infer_mvbench, check_ans
import argparse
import os

from stllm.common.config import Config
from stllm.common.registry import registry
# imports modules for registration
from stllm.datasets.builders import *
from stllm.models import *
from stllm.processors import *
from stllm.runners import *
from stllm.tasks import *

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--ckpt-path", required=True, help="path to checkpoint file.")
    parser.add_argument("--anno-path", required=True, help="path to mvbench annotation.")
    parser.add_argument("--num-frames", type=int, required=False, default=100)
    parser.add_argument("--specified_item", type=str, required=False, default=None)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--system_llm", action='store_false')
    parser.add_argument("--ask_simple", action='store_true')
    return parser.parse_args()

def run_inference(args):
    """
    Run inference on a set of video files using the provided model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model

    print('Initializing Chat')
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_config.ckpt = args.ckpt_path
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    for name, para in model.named_parameters():
        para.requires_grad = False
    model.eval()
    
    all_token = ~(model_config.video_input=='mean')
    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    videos_len = []
    dataset = MVBench_dataset(args.anno_path, num_segments=args.num_frames, resolution=224, specified_item = args.specified_item)
    for example in tqdm(dataset):
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1

        pred = infer_mvbench(
            model,example, 
            system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n",
            question_prompt="\nOnly give the best option.",
            answer_prompt="Best option:(",
            return_prompt='(',
            system_llm=args.system_llm,
            all_token=all_token,
            ask_simple=args.ask_simple,
        )

        gt = example['answer']
        if args.specified_item:
            res_list.append({
                'video_path': example['video_path'],
                'question': example['question'],
                'pred': pred,
                'gt': gt,
            })
        else:
            res_list.append({
                'pred': pred,
                'gt': gt
            })
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1
        print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
        print(f"Total Acc: {correct / total * 100 :.2f}%")
        print('-' * 30, task_type, '-' * 30)
    acc_dict['Total Acc'] = f"{correct / total * 100 :.2f}%"
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as f:
        json.dump({
            "acc_dict": acc_dict,
            "res_list": res_list
        }, f)

              


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
