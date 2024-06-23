# Preparation For Training or Inference

## 1. Prepare the Pretrained Weights
Although some weights can be downloaded dynamically at runtime, it is recommended to pre-download them for speeding up each run.

#### Pre-trained Image Encoder (EVA ViT-g)
```
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
```
the path of image encoder weight can be modified [here](stllm/models/eva_vit.py#L433).

#### Pre-trained Q-Former and Linear Projection
```
# InstructBLIP (recommended)
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth
```
```
# MiniGPT4
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth
wget https://huggingface.co/Vision-CAIR/MiniGPT-4/blob/main/pretrained_minigpt4.pth
```
the path of Q-Former and Linear Weight can be modified in `q_former_model` and `ckpt` in each config [here](config).

#### Prepare Vriptor-STLLM Weights / ST-LLM Weights / Vicuna Weights
- From Vriptor-STLLM pretrained weights **(inference on Vriptor-STLLM)**
Please download the weights from [Vriptor-STLLM](https://huggingface.co/Mutonix/Vriptor-STLLM).

- From ST-LLM pretrained weights **(training Vriptor-STLLM)**
Please download the weights from [ST-LLM](https://huggingface.co/farewellthree/ST_LLM_weight/tree/main/conversation_weight).

- From Vicuna weights **(training from scratch)**
Please first follow the [instructions](https://github.com/lm-sys/FastChat) to prepare Vicuna v1.1 (for InstructBLIP) or Vicuna v1.0 (for MiniGPT4). 
Then modify the ```llama_model``` in each config [here](config) to the folder that contains Vicuna weights.


## 2. Prepare the Data 
### Data for Inferring Vriptor-STLLM
You can have a try using the videos in [this folder](https://github.com/mutonix/Vript/tree/main/vriptor/video_examples).

### Data for Training Vriptor-STLLM
1. Download the Vript dataset from [Huggingface](https://huggingface.co/datasets/Mutonix/Vript) or [ModelScope](https://modelscope.cn/datasets/mutonix/Vript).

2. Extract the `zip` files in the `vript_long_videos_clips` folder and `vript_short_videos_clips` folder. Put the extracted videos to the `training_data/videos` folder. Extract the captions in the `vript_captions` folder and put them to the `training_data/vript_captions` folder.

3. Run the following script to generate the training data. You can directly run the commands to have a try using the provided data in the `training_data` folder.
```
### Stage 1 ###
# Generate the training data for Vriptor-STLLM training stage 1 (Whole Video)
build_vript_training_data/build_training_vript_stage1_single.py \
    --video_folder training_data/videos \
    --caption_dir training_data/vript_captions \
    --output_dir training_data/vriptor

# Generate the training data for Vriptor-STLLM training stage 1 (Multiple scenes)
build_vript_training_data/build_training_vript_stage1_concat.py \
    --video_folder training_data/videos \
    --caption_dir training_data/vript_captions \
    --output_dir training_data/vriptor

### Or Stage 2 ###
# Generate the training data for Vriptor-STLLM training stage 2 (Whole Video)
build_vript_training_data/build_training_vript_stage2_single.py \
    --video_folder training_data/videos \
    --caption_dir training_data/vript_captions \
    --output_dir training_data/vriptor

# Generate the training data for Vriptor-STLLM training stage 2 (Multiple scenes)
build_vript_training_data/build_training_vript_stage2_concat.py \
    --video_folder training_data/videos \
    --caption_dir training_data/vript_captions \
    --output_dir training_data/vriptor
```

## 3. Set Up the Environment
```
conda create -n vriptor python=3.8
conda activate vriptor
pip install -r requirements.txt
```

## 4. Run the Code
### Inferring Vriptor-STLLM
```
python demo.py \
    --video-path video_examples/emoji.mp4 \
    --cfg-path config/vriptor_stllm_stage2.yaml \
    --gpu-id 0 \
    --ckpt-path model_weights/vriptor_stllm_stage2 
```

### Training Vriptor-STLLM
```
torchrun --nproc_per_node 8 train_hf.py \
    --cfg-path config/vriptor_stllm_stage1.yaml 
    # or config/vriptor_stllm_stage2.yaml
```

