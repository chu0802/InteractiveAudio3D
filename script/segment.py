import argparse
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from src.sam.mask_processor import mask_processor
import cv2
from pathlib import Path
from tqdm import tqdm

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=Path, required=True)
    parser.add_argument('--sam_ckpt_path', type=str, default="../langsplat/ckpts/sam_vit_h_4b8939.pth")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument("--output_dir", type=Path, default="assets")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--image_id", type=int, default=0)
    parser.add_argument("--rim_size", type=int, default=10)
    
    return parser.parse_args()

def main(args):
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt_path).to(args.device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        min_mask_region_area=100,
    )

    image_path = args.dataset_path / args.mode / "ours_30000" / "renders" / f"{args.image_id:05d}.png"
    
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    output_dir = args.output_dir / args.dataset_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_images, mask_images, seg_map = mask_processor(image, mask_generator, output_dir, rim_size=args.rim_size)

    # for image_path in tqdm(sorted(image_dir.iterdir()), desc=f"Processing {args.mode} images"):
    #     image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    #     output_dir = args.output_dir / args.dataset_path.stem / args.mode / "intermediate_results" / image_path.stem
    #     output_dir.mkdir(parents=True, exist_ok=True)
    #     cv2.imwrite((output_dir / "original_image.png").as_posix(), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    #     seg_images, mask_images, seg_map = mask_processor(image, mask_generator, output_dir)

if __name__ == '__main__':
    args = arg_parse()
    main(args)
