import torch
from diffusers import LTXConditionPipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image, load_video
from diffusers.hooks import apply_group_offloading

import argparse
from pathlib import Path

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Using hand to squeeze the Sensodyne toothpaste tube")
    parser.add_argument("--image", type=Path, default=Path("test.png"))
    parser.add_argument("--seed", type=int, default=1102)
    parser.add_argument("--output", type=Path, default=Path("output.mp4"))
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()
    return args

def main(args):
    pipe = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.7-distilled", torch_dtype=torch.bfloat16)

    onload_device = torch.device("cuda")
    offload_device = torch.device("cpu")
    pipe.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", use_stream=True)
    apply_group_offloading(pipe.text_encoder, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=2)
    apply_group_offloading(pipe.vae, onload_device=onload_device, offload_type="leaf_level")

    image = load_image(args.image.as_posix())
    video = load_video(export_to_video([image])) # compress the image using video compression as the model was trained on videos
    condition1 = LTXVideoCondition(video=video, frame_index=0)

    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    expected_height, expected_width = 512, 512
    num_frames = 121
    
    for i in range(args.num_samples):
        seed = args.seed + i
        with torch.no_grad():
            video = pipe(
                conditions=[condition1],
                prompt=args.prompt,
                negative_prompt=negative_prompt,
                width=expected_width,
                height=expected_height,
                num_frames=num_frames,
                num_inference_steps=7,
                decode_timestep = 0.05,
                guidance_scale=1.0,
                decode_noise_scale = 0.025,
                output_type="pil",
                generator=torch.Generator(device=onload_device).manual_seed(seed),
            ).frames[0]

        output_dir = args.output.parent
        export_to_video(video, output_dir / f"{args.output.stem}_{seed}.mp4", fps=24)

if __name__ == "__main__":
    args = arg_parse()
    main(args)
