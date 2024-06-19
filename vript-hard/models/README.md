### README

We provide scripts to evaluate various existing models on the Vript-Hard benchmark.

For example, using Videochat2, first, ensure you have set up the Videochat2 codebase and environment:
[Videochat2 Repository](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2)

Next, execute the following command within the ./video_chat2 directory:
```
python ./videochat2_vriptRR.py \
    --model_path your_path_to_model/videochat2_7b_stage3.pth \
    --rr_data_path your_path_to_rr_dataset \
    --output_filename_video your_path_to_output/RR_video_output.csv \
    --output_filename_clip your_path_to_output/RR_clip_output.csv
```
