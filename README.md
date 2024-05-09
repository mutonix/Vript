<p align="center">
<img width="500px" alt="Vript" src="assets/Vript-title.png">
</p>


# 🎬 Vript: Refine Video Captioning into Video Scripting
---

## Updates
- 🔥 **2024-05-09**: We release **Vriptor-stllm**, a superior video captioning model trained upon [Vript](https://huggingface.co/datasets/Mutonix/Vript/) dataset based on [ST-LLM](https://github.com/TencentARC/ST-LLM/) model. **You can now try this model on the [🤗 Space](https://huggingface.co/spaces/Mutonix/Vriptor-stllm).** We will soon release the model weight.


- **2024-05-06**: We re-annotate the Vript-CAP for a more detaild captioning benchmark. In Vript-CAP, we describe the scene twice as detailed as possible with ~250 words, which is **20x** longer than the existing video captioning benchmarks. e.g., MSVD, MSR-VTT. 

- 🔥 **2024-04-15**: We release the **Vript** dataset and **Vript HardBench** benchmark. Both videos and annotations are **directly** available on [🤗](https://huggingface.co/collections/Mutonix/vript-datasets-661a80dc080a813b6ea95b50). We offer both untrimmed videos and video clips in 720p (higher resolutions may be available later).

- [WIP] We are evaluating various models on the Vript HardBench benchmark and will release the leaderboard soon.

## Introduction
We construct a **fine-grained** video-text dataset with 12K annotated YouTube videos **(~400k clips)**. The annotation of this dataset is inspired by the video script. If we want to make a video, we have to first write a script to organize how to shoot the scenes in the videos. To shoot a scene, we need to decide the content, shot type (medium shot, close-up, etc), and how the camera moves (panning, tilting, etc). Therefore, we extend video captioning to video scripting by annotating the videos in the format of video scripts. Different from the previous video-text datasets, we densely annotate the entire videos without discarding any scenes and each scene has a caption with **~145** words. Besides the vision modality, we transcribe the voice-over into text and put it along with the video title to give more background information for annotating the videos.

There are some takeaways from the Vript dataset:

1) **Fine-grained**: The Vript dataset is annotated with detailed captions of ~145 words for each scene, which contain the shot type, camera movement, content, and scene title.

2) **Dense Annotation**: The Vript dataset is densely annotated with detailed captions for all scenes in the entire videos. Each video has ~40 scenes and lasts for ~6m on average (max 3h, min 5s). The total duration of the videos is ~1.3Kh.

3) **High-quality**: The Vript dataset is annotated by GPT-4V/Claude 3 Sonnet. We find that GPT-4V has the best performance in generating detailed captions for videos and Claude 3 Sonnet has a looser constraint on the video content so that it can caption some scenes that GPT-4V cannot.

4) **High-resolution & Diverse Aspect Ratios & Open Domain**: The Vript dataset contains both long videos from YouTube and short videos from YouTube Shorts and TikTok. The raw videos vary in 720p to 2K resolution.


In addition, we propose **Vript HardBench**, a new benchmark consisting of three challenging video understanding tasks **that much harder than the existing video understanding benchmarks.**:

1) **Vript-CAP (Caption)**: A benchmark with very detailed captions rather than short captions. Each caption has ~250 words on average, which is longer than Vript train captions and **25x** longer than the existing video captioning benchmarks. e.g., MSVD, MSR-VTT. Every details in Vript-CAP are carefully checked.

2) **Vript-RR (Retrieve then Reason)**: A video reasoning benchmark by first giving a detailed description of the scene as a hint and then asking questions about details in the scene. 

3) **Vript-ERO (Event Re-ordering)**: A benchmark that tests the temporal understanding by offering the descriptions of scenes located in two/four different timelines of the same video, and asks the model to give the right temporal order of the scenes.

$\quad$

<p align="center">
<img src="assets/Vript-overview_00.png" width="800">  
</p>

$\quad$

<p align="center">
<img src="assets/Vript-bench_00.png" width="800">
</p>



## Getting Started
You can download the [Vript dataset](https://huggingface.co/datasets/Mutonix/Vript/) and Vript HardBench validation set ([Vript-CAP](https://huggingface.co/datasets/Mutonix/Vript-CAP/), [Vript-RR](https://huggingface.co/datasets/Mutonix/Vript-RR/), [Vript-ERO](https://huggingface.co/datasets/Mutonix/Vript-ERO/)) on the Huggingface.
**By downloading these datasets, you agree to the terms of the [License](#License).**

The captions of the videos in the Vript dataset are structured as follows:
```
{
    "meta": {
        "video_id": "339dXVNQXac",
        "video_title": "...",
        "num_clips": ...,
        "integrity": true, 
    },
    "data": {
            "339dXVNQXac-Scene-001": {
                "video_id": "339dXVNQXac",
                "clip_id": "339dXVNQXac-Scene-001",
                "video_title": "...",
                "caption":{
                    "shot_type": "...",
                    "camera_movement": "...",
                    "content": "...",
                    "scene_title": "...",
                },
                "voiceover": ["..."],
            },
            "339dXVNQXac-Scene-002": {
                ...
            }
        }
}
```
- `video_id`: The ID of the video from YouTube.
- `video_title`: The title of the video.
- `num_clips`: The number of clips in the video. If the `integrity` is `false`, some clips may not be captioned.
- `integrity`: Whether all clips are captioned.
- `clip_id`: The ID of the clip in the video, which is the concatenation of the `video_id` and the scene number.
- `caption`: The caption of the scene, including the shot type, camera movement, content, and scene title.
- `voiceover`: The transcription of the voice-over in the scene.


More details about the dataset and benchmark can be found in the [DATA.md](DATA.md).

## How to use Vript HardBench

### Vript-RR (Retrieve then Reason)
<p align="center">
<img src="assets/Vript-RR_01.png" width="800">
</p>

#### Input of Vript-RR
There are two ways to evaluate on the Vript-RR benchmark:

1. `Vript-RR-full` Task: 
```
Input: `full video` + `hint` + `question`
```
We input the whole video and the hint to the model and ask the question. The model can first locate the scene using the hint and then answer the question, which is more challenging.

2. `Vript-RR-clip` Task:
```
Input: `clip` + `hint` + `question`
```
We input the related scene instead of the whole video and the hint to the model and ask the question. The model can answer the question based on the related scene, which is more easy.

#### Output of Vript-RR
There are also two ways to evaluate the output of the Vript-RR benchmark:
1. Multiple Choices.
2. Open-ended. (The evaluation of open-ended questions based on GPT-4 evaluation is still in progress.)

#### Categories in Vript-RR
<p align="center">
<img src="assets/Vript-RR_00.png" width="500">
</p>


## Annotation Details
### Prompt
```python
title = video_info["title"]
if title:
    title_instruction = f'from a video titled "{title}" '
else:
    title_instruction = ""

for scene in scene_dir:
    content = []
    voiceover_instruction = ""
    if scene in voiceovers_dict and 'short' not in args.video_dir:
        voiceover = voiceovers_dict[scene]
        if voiceover['full_text'].strip():
            voiceover_text = voiceover['full_text'].strip() 
            content.append({"type": "text", "text": f"Voice-over: {voiceover_text}\nVideo frames:"})
            voiceover_instruction = "voice-over and "
        else:
            voiceover_text = ""
    else:
        voiceover_text = ""

    scene_imgs = os.listdir(os.path.join(args.video_dir, vdir, scene))
    scene_imgs = [i for i in scene_imgs if i.endswith(".jpg")]
    scene_imgs = sorted(scene_imgs, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    encoded_scene_imgs = []
    for scene_img in scene_imgs:
        encoded_scene_img = encode_image(os.path.join(args.video_dir, vdir, scene, scene_img))
        encoded_scene_imgs.append(encoded_scene_img)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_scene_img}", "detail": "low"}})

            content.append({"type": "text", "text": f"""
Based on the {voiceover_instruction}successive frames {title_instruction}above, please describe:
1) the shot type (15 words)
2) the camera movement (15 words)
3) what is happening as detailed as possible (e.g. plots, characters' actions, environment, light, all objects, what they look like, colors, etc.) (150 words)   
4) Summarize the content to title the scene (10 words)
Directly return in the json format like this:
{{"shot_type": "...", "camera_movement": "...", "content": "...", "scene_title": "..."}}. Do not describe the frames individually but the whole clip.
"""})
            
    messages=[
        {
            "role": "system",
            "content": "You are an excellent video director that can help me analyze the given video clip."
        },
        {
            "role": "user",
            "content": content
        }
    ]
```

### Sampling Strategy
```python
duration_time = total_frames / fps

# if duration < 6s
if duration_time < 6:
    # extract 20% 50% 80% frame
# if duration < 30s
elif duration_time < 30:
    # extract 15% 40% 60% 85% frame
else:
    # extract 15% 30% 50% 70% 85% frame
```

Thanks to [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) for dividing the video into scenes.

## License
By downloading or using the data or model, you understand, acknowledge, and agree to all the terms in the following agreement.

- ACADEMIC USE ONLY

Any content from Vript/Vript HardBench dataset and Vriptor model is available for academic research purposes only. You agree not to reproduce, duplicate, copy, trade, or exploit for any commercial purposes

- NO DISTRIBUTION

Respect the privacy of personal information of the original source. Without the permission of the copyright owner, you are not allowed to perform any form of broadcasting, modification or any other similar behavior to the data set content.

- RESTRICTION AND LIMITATION OF LIABILITY

In no event shall we be liable for any other damages whatsoever arising out of the use of, or inability to use this dataset and its associated software, even if we have been advised of the possibility of such damages.

- DISCLAIMER

You are solely responsible for legal liability arising from your improper use of the dataset content. We reserve the right to terminate your access to the dataset at any time. You should delete the Vript/Vript HardBench dataset or Vriptor model if required.

This license is modified from the [HD-VG-100M](https://github.com/daooshee/HD-VG-130M) license.


<!-- ## Citation
```
``` -->

## Contact
**Dongjie Yang**: [djyang.tony@sjtu.edu.cn](djyang.tony@sjtu.edu.cn)