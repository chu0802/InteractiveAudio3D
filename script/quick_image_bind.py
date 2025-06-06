from src.imagebind.models import imagebind_model
from src.imagebind.models.imagebind_model import ModalityType
from src.imagebind import data as imagebind_data
import torch
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_path", type=str, required=True)
    parser.add_argument("-a", "--audio_path", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to("cuda")

    video_path = [args.video_path]
    audio_path = [args.audio_path]
    
    video_data = imagebind_data.load_and_transform_video_data(video_path, "cuda")
    audio_data = imagebind_data.load_and_transform_audio_data(audio_path, "cuda")
    
    inputs = {
        ModalityType.VISION: video_data,
        ModalityType.AUDIO: audio_data,
    }
    
    with torch.no_grad():
        embeddings = model(inputs)
    
    normalized_audio = embeddings[ModalityType.AUDIO] / embeddings[ModalityType.AUDIO].norm(dim=1, keepdim=True)
    normalized_vision = embeddings[ModalityType.VISION] / embeddings[ModalityType.VISION].norm(dim=1, keepdim=True)
    scores = normalized_audio @ normalized_vision.T

    print(scores)
