import argparse
from pathlib import Path
from transformers import ClapProcessor, ClapModel
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio
import numpy as np

def load_wav_mono(path, target_sr=48000):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    if wav.size(0) > 1:
        wav = wav.mean(dim=0)
    return wav.to(torch.float32).contiguous()

@torch.no_grad()
def clap_embed(waveform, model, processor, device, sampling_rate=48000):
    if waveform.dtype != torch.float32:
        waveform = waveform.float()
    inputs = processor(audios=[waveform.cpu().numpy()], sampling_rate=sampling_rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.get_audio_features(**inputs)
    emb = out.audio_embeds if hasattr(out, "audio_embeds") else out
    emb = F.normalize(emb.float(), dim=-1)
    return emb.squeeze(0)

def get_clap_features(file_paths, model, processor, device):
    embeddings = []
    for path in tqdm(file_paths, total=len(file_paths)):
        wav = load_wav_mono(path)
        emb = clap_embed(wav, model, processor, device)
        embeddings.append(emb)
    return torch.stack(embeddings, dim=0)


def compute_knn_smoothed(
    embeddings,
    overall_scores,
    k=99,
    temp=1.0,
    beta=1.0,
) -> None:
    scores = torch.tensor(overall_scores, dtype=torch.float32, device=embeddings.device)

    dist = 1.0 - torch.matmul(embeddings, embeddings.T)

    N = dist.shape[0]
    k = min(k, N - 1)
    knn_idx = torch.zeros((N, k), dtype=torch.long, device=embeddings.device)

    for i in range(N):
        sorted_idx = torch.argsort(dist[i])
        sorted_idx = sorted_idx[sorted_idx != i]
        knn_idx[i] = sorted_idx[:k]

    d_knn = torch.take_along_dim(dist, knn_idx, dim=1)
    sigma2 = torch.median(d_knn, dim=1, keepdim=True).values * temp

    w = torch.exp(-d_knn / (sigma2 + 1e-12))
    w_sum = w.sum(dim=1, keepdim=True) + 1e-12
    smoothed = (w * scores[knn_idx]).sum(dim=1, keepdim=True) / w_sum
    smoothed = smoothed.squeeze(1)
    
    smoothed = smoothed * beta + scores * (1 - beta)

    smoothed = smoothed + (scores.mean() - smoothed.mean())
    
    return smoothed


def main(args):
    
    with open(args.dataset_dir / args.scene_name / "simplified_hands_only_interaction_results.json", "r") as f:
        info = json.load(f)
        
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    model = ClapModel.from_pretrained("laion/clap-htsat-fused").to("cuda").eval()

    for img_id, obj_info in info.items():
        obj_name = obj_info["object_name"]
        audio_dir = args.output_dir / args.scene_name / f"{obj_name.replace(' ', '_')}" / f"iter{args.iter}"
        if args.target_obj and obj_name.replace(" ", "_") != args.target_obj.replace(" ", "_"):
            continue
        
        # load reward file
        with open(audio_dir / f"rewards.json", "r") as f:
            rewards = json.load(f)
            
        for interaction_info in obj_info["interactions"]:
            prompt = interaction_info["description"].replace(" ", "_")
            
            # filter keys in reward that contains the prompt
            file_paths = [key for key in rewards.keys() if prompt in key]
            
            embeddings = get_clap_features(file_paths, model, processor, "cuda")
            overall_scores = [rewards[key]["scores"]["overall"] for key in file_paths]
            
            smoothed_scores = compute_knn_smoothed(embeddings, overall_scores)
            
            # update rewards
            for file_path, score in zip(file_paths, smoothed_scores):
                rewards[file_path]["scores"]["smoothed"] = score.item()
        
        # save rewards
        with open(audio_dir / "smoothed_rewards.json", "w") as f:
            json.dump(rewards, f, indent=4)

            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=Path, default=Path("logs/audios"))
    parser.add_argument("-d", "--dataset_dir", type=Path, default=Path("datasets"))
    parser.add_argument("-s", "--scene_name", type=str, default="0118_bathroom")
    parser.add_argument("-t", "--target_obj", type=str, default=None)
    parser.add_argument("-i", "--iter", type=int, default=0)
    args = parser.parse_args()
    main(args)
