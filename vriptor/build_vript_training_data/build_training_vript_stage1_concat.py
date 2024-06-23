import os
import json
import random
import tqdm

random.seed(42)

def merge_voiceovers(voiceovers):
    merged_voiceover = []
    wait_merge_voiceover = []
    for v in voiceovers:
        wait_merge_voiceover.append(v)
        text = " ".join([w['text'] for w in wait_merge_voiceover])
        if len(text.split()) > 15:
            merged_voiceover.append({
                "span": [wait_merge_voiceover[0]['span'][0], wait_merge_voiceover[-1]['span'][1]],
                "text": " ".join([w['text'] for w in wait_merge_voiceover])
            })
            wait_merge_voiceover = []

    if wait_merge_voiceover:
        merged_voiceover.append({
            "span": [wait_merge_voiceover[0]['span'][0], wait_merge_voiceover[-1]['span'][1]],
            "text": " ".join([w['text'] for w in wait_merge_voiceover])
        })

    return merged_voiceover

def build_training_vript(args):
    video_folder = args.video_folder
    caption_dir = args.caption_dir

    caption_files = os.listdir(caption_dir)
    gather_caption_files = []
    for cap_files in caption_files:
        gather_caption_files.extend([os.path.join(caption_dir, cap_files, f) for f in os.listdir(os.path.join(caption_dir, cap_files)) if f.endswith("_caption.json")])

    print(gather_caption_files[:10])

    video_dirs = os.listdir(video_folder)
    video_dict = {}
    for vdir in video_dirs:
        if not os.path.isdir(os.path.join(video_folder, vdir)):
            continue

        video_files = os.listdir(os.path.join(video_folder, vdir))
        video_files = [f for f in video_files if os.path.isdir(os.path.join(video_folder, vdir, f))]
        for vfile in video_files:
            video_id = vfile.split("/")[-1]
            video_dict[vfile] = os.path.join(vdir, vfile)

    voice_over_num = 0
    training_captions = []
    scene_len_weights = [1, 2, 3, 4, 3, 2]
    for cfile in tqdm.tqdm(gather_caption_files):
        video_id = cfile.split("/")[-1].split("_caption.json")[0]
        video_file_path = video_dict[video_id]
        caption = json.load(open(os.path.join(cfile)))

        cap_keys = sorted(caption['data'].keys(), key=lambda x: int(x.split("-Scene-")[-1]))

        num_scenes = len(cap_keys)
        num_iter = min(30, int(num_scenes // 1.5))

        start_end_scene_idx = []
        for i in range(num_iter):
            start_scene_idx = random.randint(0, len(cap_keys) -2)
            scene_len = random.choices([1, 2, 3, 4, 5, 6], weights=scene_len_weights)[0]
            if start_scene_idx + scene_len > len(cap_keys):
                continue

            start_number = int(cap_keys[start_scene_idx].split('-Scene-')[-1])
            end_number = int(cap_keys[start_scene_idx + scene_len - 1].split('-Scene-')[-1])
            if end_number - start_number + 1 != scene_len:
                continue

            end_scene_idx = start_scene_idx + scene_len
            start_end_scene_idx.append((start_scene_idx, end_scene_idx))

        if os.path.exists(os.path.join(video_folder, video_dict[video_id], f"{video_id}_asr.jsonl")):
            voiceover_list = [json.loads(l) for l in open(os.path.join(video_folder, video_dict[video_id], f"{video_id}_asr.jsonl"))]
        else:
            continue
        voiceover_dict = {v['clip_id']: v for v in voiceover_list}

        for start_scene_idx, end_scene_idx in start_end_scene_idx:
            voiceover = ""
            end_time_of_prev_scene = 0
            try:
                prev_video_duration = 0
                durations = []
                for k in cap_keys[start_scene_idx:end_scene_idx]:
                    voiceovers = voiceover_dict[caption['data'][k]['clip_id']]['scene_text']
                    # voiceovers = merge_voiceovers(voiceovers)
                    duration = f"[{round(float(prev_video_duration), 1)}, {round(float(prev_video_duration + voiceover_dict[caption['data'][k]['clip_id']]['duration']), 1)}]"
                    durations.append(duration)
                    prev_video_duration += voiceover_dict[caption['data'][k]['clip_id']]['duration']
                    for v in voiceovers:
                        span = v['span']
                        span = (span[0] + end_time_of_prev_scene, span[1] + end_time_of_prev_scene)
                        voiceover += f"[{round(float(span[0]), 1)}, {round(float(span[1]), 1)}] " + v['text'].strip() + "\n"
                    end_time_of_prev_scene = span[1]
            except:
                continue

            prev_video_duration = round(max(prev_video_duration, float(durations[-1].split(", ")[1].strip("]"))), 1)
            durations[-1] = durations[-1].split(", ")[0] + ", " + f"{prev_video_duration}]"


            caption_content = ""
            for i, k in enumerate(cap_keys[start_scene_idx:end_scene_idx]):
                caption_content += f"[Scene {i+1}: {caption['data'][k]['caption']['scene_title']}](Duration: {durations[i]})\n{caption['data'][k]['caption']['content']}\n\n"
            caption_content = caption_content.strip()

            instruct = ""
            sample_rate = round(prev_video_duration  / 64, 2)
            if voiceover and (("voice" in caption_content.lower() and "over" in caption_content.lower()) or "voicing" in caption_content.lower() or "narrat" in caption_content.lower() or "speak" in caption_content.lower() or "talk" in caption_content.lower() or "say" in caption_content.lower()):
                instruct = f"""Voiceover:"{voiceover}". Based on the video and voiceover, describe the video scene by scene in detail (Duration: [0.0, {round(prev_video_duration, 1)}]second, Sampling Rate: {sample_rate}second/frame)."""
                voice_over_num += 1
            else:
                instruct = f"Describe the video scene by scene in detail (Duration: [0.0, {round(prev_video_duration, 1)}]second, Sampling Rate: {sample_rate}second/frame)."

            training_captions.append({
                "video": [
                    os.path.join(video_file_path, f"{caption['data'][k]['clip_id']}.mp4") for k in cap_keys[start_scene_idx:end_scene_idx]
                ],
                "QA":[
                    {"i": instruct, "q": "", "a": caption_content},
                ]
            })

    random.shuffle(training_captions)
    json.dump(training_captions, open(os.path.join(args.output_dir, "vript_stage1_concat.json"), "w"), indent=2)
    print("Number of training samples:", len(training_captions))
    print("Number of voiceover samples:", voice_over_num)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, default="training_data/videos")
    parser.add_argument("--caption_dir", type=str, default="training_data/vript_captions")
    parser.add_argument("--output_dir", type=str, default="training_data/vriptor")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    build_training_vript(args)