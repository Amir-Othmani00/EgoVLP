import numpy as np
import json
from collections import defaultdict
import os
import argparse

parser = argparse.ArgumentParser(description="Format visual features for matching.")
parser.add_argument('--annotations', type=str, default='annotations/annotation_json/complete_step_annotations.json', help='Path to complete annotations json')
parser.add_argument('--step_metadata', type=str, default='visual_features/best-results-so-far.json', help='Path to step metadata json')
parser.add_argument('--embeddings', type=str, default='visual_features/best-hiero-embeddings-so-far.npz', help='Path to feature embeddings npz')
parser.add_argument('--out_npz', type=str, default='visual_features/best_hiero_step_embeddings_256.npz', help='Output npz path')
parser.add_argument('--out_json', type=str, default='visual_features/best_visual_features_mapping.json', help='Output json path')
args = parser.parse_args()

print(f"Loading annotations for task mappings from {args.annotations}...")
with open(args.annotations, 'r') as f:
    annotations = json.load(f)

print(f"Loading step metadata from {args.step_metadata}...")
with open(args.step_metadata, 'r') as f:
    best_results = json.load(f)

print(f"Loading feature embeddings from {args.embeddings}...")
hiero_emb = np.load(args.embeddings)

task_to_features = defaultdict(list)
task_step_metadata = defaultdict(list)
task_stats = defaultdict(lambda: {"total_steps": 0, "correct_steps": 0, "incorrect_steps": 0})
video_to_task = {}
video_to_label = {}

video_idx_counter = 0

for rec_id in hiero_emb.files:
    if rec_id not in annotations:
        print(f"Warning: {rec_id} not in annotations!")
        continue
        
    activity_str = annotations[rec_id]['activity_name']
    task_name = activity_str.lower().replace(' ', '')
    label = 1 if any(s.get('has_errors', False) for s in annotations[rec_id].get('steps', [])) else 0
    
    vid_idx = video_idx_counter
    video_idx_counter += 1
    
    video_to_task[str(vid_idx)] = task_name
    video_to_label[str(vid_idx)] = label
    
    feats = hiero_emb[rec_id] # (num_steps, 256)
    if feats.ndim == 1:
        feats = feats[np.newaxis, :]
        
    task_to_features[task_name].append(feats)
    
    step_info_list = best_results.get(rec_id, {}).get('steps', [])
    
    for step_idx in range(feats.shape[0]):
        step_meta = {
            "recording_id": rec_id,
            "video_idx": vid_idx,
            "step_idx_in_video": step_idx,
            "label": label
        }
        
        # Add explicit fine-grained metadata like start/end times and step_id
        if step_idx < len(step_info_list):
            step_meta.update(step_info_list[step_idx])
            
        task_step_metadata[task_name].append(step_meta)
        
        task_stats[task_name]["total_steps"] += 1
        if label == 0:
            task_stats[task_name]["correct_steps"] += 1
        else:
            task_stats[task_name]["incorrect_steps"] += 1

reorganized_features = {}
for task_name, feats_list in task_to_features.items():
    reorganized_features[task_name] = np.concatenate(feats_list, axis=0)
    print(f"Task {task_name}: {reorganized_features[task_name].shape[0]} total steps")

# Output files 
npz_out = args.out_npz
np.savez(npz_out, **reorganized_features)
print(f"Saved {npz_out}")

json_out = args.out_json
metadata = {
    "video_to_task": video_to_task,
    "video_to_label": video_to_label,
    "task_step_metadata": dict(task_step_metadata),
    "task_stats": dict(task_stats)
}
with open(json_out, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Saved {json_out}")

print("Done grouping embeddings by task.")
