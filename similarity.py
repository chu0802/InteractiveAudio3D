from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scene_name", type=str, default="0118_bathroom")
parser.add_argument("--image_id", type=int, default=0)
args = parser.parse_args()


image_properties_path = Path("output") / args.scene_name / f"{args.image_id:03d}" / "image_properties.json"
interactions_path = Path("output") / args.scene_name / f"{args.image_id:03d}" / "interactions.json"


# audio_properties_path = Path("output/0118_bathroom/000/dropping_Dial_hand_soap/audio_understanding_candidates_ver3.json")
# audio_properties_path = Path("output/0118_bathroom/000/dropping_the_full_clear_plastic_hand_soap_dispenser_onto_the_floor/audio_understanding_candidates.json")
# audio_properties_path = Path("output/0118_bathroom/000/dropping_the_full_clear_plastic_hand_soap_dispenser_onto_the_floor/audio_understanding_candidates.json")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to("cuda")

with open(image_properties_path, "r") as f:
    image_properties = json.load(f)

with open(interactions_path, "r") as f:
    interactions = json.load(f)

for interaction in interactions["interactions"]:
    query_action = interaction.split(" ")[0]
    
    audio_properties_path = interactions_path.parent / f"{'_'.join(interaction.split(' '))}/audio_understanding_candidates.json"

    with open(audio_properties_path, "r") as f:
        audio_properties = json.load(f)

    queried_action_emb = model.encode(query_action, convert_to_tensor=True, normalize_embeddings=True)
    quried_objects_emb = model.encode(image_properties["object_name"], convert_to_tensor=True, normalize_embeddings=True)
    queried_properties_emb = model.encode(image_properties["properties"], convert_to_tensor=True, normalize_embeddings=True)

    results = {}
    for seed, audio_property in audio_properties.items():
        candidate_actions_emb = model.encode(audio_property["potential_actions"], convert_to_tensor=True, normalize_embeddings=True)
        candidate_objects_emb = model.encode(audio_property["potential_objects"], convert_to_tensor=True, normalize_embeddings=True)
        candidate_properties_emb = model.encode(audio_property["potential_materials_and_properties"], convert_to_tensor=True, normalize_embeddings=True)
        
        cos_sim_action = torch.einsum("k,jk->j", queried_action_emb, candidate_actions_emb)
        cos_sim_object = torch.einsum("k,jk->j", quried_objects_emb, candidate_objects_emb)
        cos_sim_property = torch.einsum("ik,jk->ij", queried_properties_emb, candidate_properties_emb).max(dim=1).values

        avg_action_sim = cos_sim_action.max()
        avg_object_sim = cos_sim_object.max()
        avg_property_sim = cos_sim_property.mean()
        print(f"Seed: {seed}, Total score: {(avg_action_sim + avg_object_sim + avg_property_sim).item()}")

        results[seed] = {
            "cos_sim_action": cos_sim_action.detach().cpu().numpy().tolist(),
            "cos_sim_object": cos_sim_object.detach().cpu().numpy().tolist(),
            "cos_sim_property": cos_sim_property.detach().cpu().numpy().tolist(),
            "avg_action_sim": avg_action_sim.item(),
            "avg_object_sim": avg_object_sim.item(),
            "avg_property_sim": avg_property_sim.item(),
            "total_score": (avg_action_sim + avg_object_sim + avg_property_sim).item()
        }
        output_dir = audio_properties_path.parent
        with open(output_dir / f"similarity_results_all_max.json", "w") as f:
            json.dump(results, f, indent=4)
