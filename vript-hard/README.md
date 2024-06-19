## How to evaluate on Vript-Hard
### Get the prediction of your model
For fair comparison, please evaluate your own model on the Vript-Hard benchmark using these evaluation prompts in [here](https://github.com/mutonix/Vript/tree/main/vript-hard/evaluation_prompts/).

The output of Vript-Hard is recommended to be in the format used in the [examples](http://github.com/mutonix/Vript/tree/main/vript-hard/evaluation_output_examples/), which is a csv file. The csv file should contain the following columns:
- `id`: The ID of the video or clip.
- `pred`: The prediction of the model.
- `gt`: Optional. If not provided, we will used the ground truth answers automatically downloaded from the Huggingface.

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
