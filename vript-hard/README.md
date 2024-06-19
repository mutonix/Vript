## How to evaluate on Vript-Hard
### Get the prediction of your model
**A quick start of evaluating VideoChat2**
For fair comparison, please evaluate your own model on the Vript-Hard benchmark using these evaluation prompts in [here](https://github.com/mutonix/Vript/tree/main/vript-hard/evaluation_prompts/).

We provide an example of evaluating the VideoChat2 model on Vript-Hard. First of all, you have to set up the Videochat2 codebase and environment following [the instructions of Videochat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2).

Next, copy our [evaluation python files](https://github.com/mutonix/Vript/tree/main/vript-hard/models/videochat2) to the `Ask-Anything/video_chat2` and run the evaluation command. For example, you can evaluate the Videochat2 model on the Vript-RR benchmark using the following commands:
```
cd Ask-Anything/video_chat2
cp /path_to_Vript/Vript/vript-hard/models/videochat2/videochat2_vriptRR.py ./

python videochat2_vriptRR.py \
    --model_path your_path_to_model/videochat2_7b_stage3.pth \
    --rr_data_path your_path_to_rr_dataset \
    --output_filename_video your_path_to_output/RR_video_output.csv \
    --output_filename_clip your_path_to_output/RR_clip_output.csv
```

In the above example, we format the prediction of the model as the one in the [output example](http://github.com/mutonix/Vript/tree/main/vript-hard/evaluation_output_examples/), which is a csv file. The csv file should contain the following columns:
- `id`: The ID of the video or clip.
- `pred`: The prediction of the model.
- `gt`: [Optional] The ground truth answers. If they are not provided, we will used the ground truth answers automatically downloaded from the Huggingface.

<!-- 
#### Vript-RR (Retrieve then Reason)
<p align="center">
<img src="assets/Vript-RR_01.png" width="800">
</p> -->

#### PS: Further illustration of evaluating on Vript-RR
1. **Input of Vript-RR**
There are two ways to evaluate on the Vript-RR benchmark:

    - `Vript-RR-whole` Task: 
    ```
    Input: `whole video` + `question` + `hint`
    ```
    We input the whole video along with the question and hint. The model can first locate the scene using the hint and then answer the question, which is more challenging.

    - `Vript-RR-clip` Task:
    ```
    Input: `clip` + `question` + `hint`
    ```
    We input the related scene instead of the whole video along with the question and hint. The model can answer the question based on the related scene, which is more easy.

2. **Output of Vript-RR**
    There are also two ways to evaluate the output of the Vript-RR benchmark:
    - Multiple Choices.
    - Open-ended. (The verification of open-ended questions based on GPT-4 evaluation can be checked in [here](http://github.com/mutonix/Vript/tree/main/vript-hard/scripts/run_verify_RR_openended.sh).)


### Verify the prediction
1. First of all, you need to install the requirements:
```
conda create -n vript python=3.8 -y
conda activate vript
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```
2. Then, you can verify the output of your model on the Vript-Hard benchmark using the scripts in [here](http://github.com/mutonix/Vript/tree/main/vript-hard/scripts).

Except for the Vript-RR open-ended verification, you can **directly** run the following commands to have a try (We have provided the examples for an easy start). For RR open-ended verification, you should configure your GPT-4-turbo API key.
```
cd vript-hard/scripts

# Verify the output of Vript-HAL
bash run_verify_HAL.sh

# Verify the output of Vript-RR (Multiple Choices)
bash run_verify_RR.sh

# Verify the output of Vript-RR (Open-ended)
bash run_verify_RR_openended.sh 

# Verify the output of Vript-ERO
bash run_verify_ERO.sh
```
