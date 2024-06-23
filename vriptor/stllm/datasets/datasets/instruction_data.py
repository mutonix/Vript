from torchvision import transforms
from torchvision.transforms import InterpolationMode

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
normalize = transforms.Normalize(mean, std)
type_transform = transforms.Lambda(lambda x: x.float().div(255.0))
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            224,
            scale=(0.5, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        #transforms.RandomHorizontalFlip(),
        type_transform,
        normalize,
    ]
)

anno_root_it = 'training_data'

# ============== pretraining datasets=================
available_corpus = dict(
    caption_vript_stage1_single=[
        f"{anno_root_it}/vript/vript_stage1_single.json", 
        f"{anno_root_it}/videos",
        "video"
    ],
    caption_vript_stage1_concat=[
        f"{anno_root_it}/vript/vript_stage1_concat.json", 
        f"{anno_root_it}/videos",
        "video"
    ],
    caption_vript_stage2_single=[
        f"{anno_root_it}/vript/vript_stage2_single.json", 
        f"{anno_root_it}/videos",
        "video"
    ],
    caption_vript_stage2_concat=[
        f"{anno_root_it}/vript/vript_stage2_concat.json", 
        f"{anno_root_it}/videos",
        "video"
    ],
)


