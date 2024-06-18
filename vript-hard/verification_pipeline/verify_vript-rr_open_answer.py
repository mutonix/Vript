# -*- encoding:utf-8 -*-

"""
API REQUEST PARALLEL PROCESSOR
"""

# imports
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
import tqdm
import datasets
import pandas as pd

from dataclasses import dataclass  # for storing API inputs, outputs, and metadata
from queue import PriorityQueue  

import random
import numpy as np

request_header = {"Content-Type": "application/json"}
request_url = {"url": "https://api.openai.com//v1/completions"}

random.seed(42)
np.random.seed(42)


class API():
    def __init__(self, current_time, api):
        self.current_time = current_time
        self.api = api

    def __lt__(self, other):
        """ 定义<比较操作符。 """
        if self.current_time == other.current_time:
            if random.random() > 0.5:
                return True
            else:
                return False
        return self.current_time < other.current_time

async def process_api_requests_from_file(
    requests_filepath: str,
    apis_pool: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
    output_path: str
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 1
    seconds_to_sleep_each_loop = 0.01  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    # api_endpoint = api_endpoint_from_url(request_url)
    api_endpoint = "chat/completions"
    current_api_record = apis_pool.get()
    api = current_api_record.api
    # request_header.update({'api-key': api['api-key']})
    # request_header = {"Authorization": f"Bearer {api-key}"}
    request_header.update({"Authorization": f"Bearer {api['api-key']}"})
    request_url.update({'url': api['request_url']})

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    status_tracker.current_api_key = api
    status_tracker.current_api_available_time = - current_api_record.current_time
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
    with open(requests_filepath, "r", encoding="utf8") as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug(f"File opened. Entering main loop")

        while True:
            # get next request (if one is not already waiting for capacity)
            if next_request is None:
                if not queue_of_requests_to_retry.empty():
                    next_request = queue_of_requests_to_retry.get_nowait()
                    task_id = next_request.task_id
                    logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
                elif file_not_finished:
                    try:
                        # get new request
                        request_json = json.loads(next(requests))
                        token_consumption = num_tokens_consumed_from_request(request_json, api_endpoint, token_encoding_name)
                        if token_consumption >= 4096:
                            token_consumption = 4096
                        task_id = next(task_id_generator)
                        next_request = APIRequest(
                            task_id=task_id,
                            request_json=request_json,
                            token_consumption=token_consumption,
                            attempts_left=max_attempts,
                        )
                        status_tracker.num_tasks_started += 1
                        status_tracker.num_tasks_in_progress += 1
                        logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                    except StopIteration:
                        # if file runs out, set flag to stop reading it
                        logging.debug("Read file exhausted")
                        file_not_finished = False

            # update available capacity
            current_time = time.time()
            seconds_since_update = current_time - last_update_time
            available_request_capacity = min(
                available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
                max_requests_per_minute,
            )
            available_token_capacity = min(
                available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
                max_tokens_per_minute,
            )
            last_update_time = current_time

            # if enough capacity available, call API
            if next_request:
                next_request_tokens = next_request.token_consumption
                if (
                    available_request_capacity >= 1
                    and available_token_capacity >= next_request_tokens
                ):
                    # update counters
                    available_request_capacity -= 1
                    available_token_capacity -= next_request_tokens
                    next_request.attempts_left -= 1

                    # call API
                    asyncio.create_task(
                        next_request.call_api(
                            request_header=request_header,
                            retry_queue=queue_of_requests_to_retry,
                            status_tracker=status_tracker,
                            apis_pool=apis_pool,
                            output_path=output_path
                        )
                    )
                    next_request = None  # reset next_request to empty

            # if all tasks are finished, break
            if status_tracker.num_tasks_in_progress == 0:
                logging.warning(f"All tasks finished. Breaking main loop.")
                break


            # main loop sleeps briefly so concurrent tasks can run
            await asyncio.sleep(seconds_to_sleep_each_loop)

            # if a rate limit error was hit recently, pause to cool down


        # after finishing, log final status
        logging.info(f"""Parallel processing complete.""")
        if status_tracker.num_tasks_failed > 0:
            logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed.")
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")


# dataclasses


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits
    time_of_last_switch_to_new_api_key: int = 0  # used to switch API keys periodically


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    result = []

    async def call_api(
        self,
        request_header: dict,
        retry_queue: asyncio.Queue,
        status_tracker: StatusTracker,
        apis_pool: PriorityQueue = None,
        output_path: str = None,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with aiohttp.ClientSession() as session:
                json_data = self.request_json['api_input']         
                logger_flag = True
                available_time = status_tracker.current_api_available_time
                while True:
                    apis_pool.put(API(available_time, status_tracker.current_api_key))
                    next_api_record = apis_pool.get()
                    status_tracker.current_api_key = next_api_record.api
                    status_tracker.current_api_available_time = next_api_record.current_time

                    if status_tracker.current_api_available_time > time.time():
                        # seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
                        # if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
                        #     remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
                            # await asyncio.sleep(remaining_seconds_to_pause)
                        await asyncio.sleep(0.1)
                        # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                        # logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")
                        if logger_flag:
                            logging.warning(f"API key {status_tracker.current_api_key['api-key']} is paused to cool down for {round(status_tracker.current_api_available_time - time.time(), 2)}s")
                            logger_flag = False
                        available_time = status_tracker.current_api_available_time
                    else:
                        request_header.update({'api-key': status_tracker.current_api_key['api-key']})
                        request_url.update({'url': status_tracker.current_api_key['request_url']})
                        # if status_tracker.current_api_key != old_api_key:
                        logging.warning(f"Request {self.task_id} is switched to API key {status_tracker.current_api_key['api-key']} to start.")
                        break

                async with session.post(
                    url=request_url['url'], headers=request_header, json=json_data
                ) as response:
                    response = await response.json()
            if "error" in response or "Error" in response:
                if "Error" in response:
                    response["error"] = response["Error"]
                task_clip_id = self.request_json["clip_id"]
                logging.warning(
                    f"Request {self.task_id}: Clip {task_clip_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                # if "rate limit" in response["error"].get("message", "").lower() or "overloaded" in response["error"].get("message", "").lower():
                if "rate limit" in response["error"].get("message", "").lower():
                    error_message = response["error"]["message"]
                    retry_seconds = int(re.search(r"Please retry after (\d+) seconds", error_message)[1])
                    retry_time = time.time() + retry_seconds + 1
                    status_tracker.current_api_available_time = retry_time
                    
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.task_id} failed after all attempts.")
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            result = post_process_gpt_response(response)
            if result is not None:
                if not os.path.exists(output_path):
                    df = pd.DataFrame(columns=["id", "judge", "model", "response"])
                    df = pd.concat([
                        df,
                        pd.DataFrame([{"id": self.request_json["clip_id"], "judge": result["result"], "model": self.request_json["model"], "response": self.request_json["response"]}])
                    ])
                    df.to_csv(output_path, index=False)
                else:
                    df = pd.read_csv(output_path)
                    df = pd.concat([
                        df,
                        pd.DataFrame([{"id": self.request_json["clip_id"], "judge": result["result"], "model": self.request_json["model"], "response": self.request_json["response"]}])
                    ])
                    df.to_csv(output_path, index=False)

                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_succeeded += 1
            else:
                if self.attempts_left:
                    retry_queue.put_nowait(self)
                else:
                    logging.error(f"Request {self.task_id} failed after all attempts.")
                    status_tracker.num_tasks_in_progress -= 1
                    status_tracker.num_tasks_failed += 1


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data, ensure_ascii=False)
    with open(filename, "a", encoding="utf8") as f:
        f.write(json_string + "\n")

def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        request_json = request_json["api_input"] # 
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    # if value is a list
                    if isinstance(value, list):
                        for v in value:
                            if v['type'] == 'text':
                                num_tokens += len(encoding.encode(v['text']))
                            else:
                                num_tokens += 85
                    else:
                        num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant

            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


def post_process_gpt_response(responses):
    for response in responses["choices"]:
        try:
            raw_chat = response["message"]["content"]
        except:
            print("ERROR parse!")
            return None

        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if response["finish_reason"] == "length":
            print("WARNING: last example is truncated")
            return None 
        
        json_chat = re.search(r"\{.*\}", raw_chat)
        
        try:
            result = eval(json_chat.group())
        except:
            print("ERROR parse!")
            return None
        
        # check the keys of the result
        if "rating" not in result:
            print("ERROR parse!")
            return None
        
        return {"result": result["rating"]}

# run script
if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model", default="gpt-4-turbo-2024-04-09")
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--num_response", type=int, default=1)  
    parser.add_argument("--max_requests_per_minute", type=int, default=50)
    parser.add_argument("--max_tokens_per_minute", type=int, default=5_000)
    parser.add_argument("--token_encoding_name", default="cl100k_base")
    parser.add_argument("--max_attempts", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--output_path", type=str, default="../evaluation_output_samples/RR/openended/scores_RR_open_verify.csv")
    parser.add_argument("--prediction_dir", type=str, default="../evaluation_output_samples/RR/openended")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--logging_level", default=logging.INFO)
    args = parser.parse_args()

    # apis = [json.loads(l) for l in open(args.api_config, "r", encoding="utf8")]
    if args.api_key is None:
        raise ValueError("Please provide an API key.")
    
    apis = [{"api-key": args.api_key, "request_url": f"https://api.openai.com{args.endpoint}"}]
    apis_pool = PriorityQueue(len(apis))
    for i, api in enumerate(apis):
        apis_pool.put(API(time.time(), api))

    request_start = time.time()
    rr_data_file = {"test": "RR_annotations.jsonl"}
    rr_dataset = datasets.load_dataset(f"mutonix/Vript-RR", data_files=rr_data_file, cache_dir=args.cache_dir)['test']
    rr_dataset_dict = {video['clip_id']: video for video in rr_dataset}
    
    result_files = os.listdir(args.prediction_dir)
    result_files = [os.path.join(args.prediction_dir, f) for f in result_files if f.endswith(".csv") and "open" in f]
    all_results = []
    for result_file in result_files:
        df = pd.read_csv(result_file)
        df = pd.concat([
            df,
            pd.Series([result_file.split("/")[-1].split(".")[0]] * len(df), name="model")
        ], axis=1)
        all_results.append(df)
    all_results = pd.concat(all_results, axis=0)
    
    cache_path = "cache_RR_verify_open.jsonl"
    if os.path.exists(cache_path):
        os.remove(cache_path)

    def process_video(row):
        clip_id = row['id']
        response = row['pred']
        if pd.isna(response) or response == "":
            return
        model = row['model']
        if isinstance(row['meta'], str):
            meta = eval(row['meta'])
            clip_id = meta['clip_id']
        data_sample = rr_dataset_dict[clip_id]
        question = data_sample['question']
        hint = data_sample['hint']
        answer = data_sample['open_answer']
        content = []

        prompt = f"""Question: {hint}\n{question}
                
AI assistant response:
{response}

Ground truth answer:
{answer}"""
        
        content.append({"type": "text", "text": prompt})
            
        messages=[
            {
                "role": "system",
                "content": "Please act as an impartial judge and check if the response provided by an AI assistant to the question match the ground truth answer. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must judge if the AI assistant's response is correct or incorrect following this json format: {\"rating\": True} or {\"rating\": False}."
            },
            {
                "role": "user",
                "content": content
            }
        ]

        decoding_args = {
            "temperature": 0.9,
            "max_tokens": args.max_tokens,
            "top_p":0.9,
            "frequency_penalty":0.2,
            "presence_penalty":0.1,
            "model":"gpt-4",
        }

        request_json = {}
        api_input = {}
        api_input.update(decoding_args)
        api_input["messages"] = messages
        api_input["model"] = args.model
        
        request_json["api_input"] = api_input
        request_json["clip_id"] = clip_id
        request_json["model"] = model
        request_json["response"] = response
        append_to_jsonl(request_json, filename=cache_path)

    # joblib.Parallel(n_jobs=60, backend='multiprocessing')(joblib.delayed(process_video)(vdir) for vdir in tqdm.tqdm(video_dirs))
    for i, row in tqdm.tqdm(all_results.iterrows()):
        if os.path.exists(args.output_path):
            df = pd.read_csv(args.output_path)
            if row['id'] in df['id'].values and row['model'] in df['model'].values:
                if not pd.isna(df[(df['id'] == row['id']) & (df['model'] == row['model'])]['judge'].values[0]):
                    continue
        
        process_video(row)

    # run script
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=cache_path,
            apis_pool=apis_pool,
            max_requests_per_minute=float(args.max_requests_per_minute),
            max_tokens_per_minute=float(args.max_tokens_per_minute),
            token_encoding_name=args.token_encoding_name,
            max_attempts=int(args.max_attempts),
            logging_level=int(args.logging_level),
            output_path=args.output_path
        )
    )
    print(f"Total time: {round(time.time() - request_start, 2)}s")
    os.remove(cache_path)




