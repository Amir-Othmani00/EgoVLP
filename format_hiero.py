import numpy as np
import json
from collections import defaultdict

print("Loading boundaries...")
with open('visual_features/hiero_step_boundaries.json', 'r') as f:
    boundaries = json.load(f)

print("Loading hiero embeddings...")
hiero_emb = np.load('visual_features/hiero_step_embeddings.npz')

task_to_features = defaultdict(list)
task_step_metadata = defaultdict(list)
task_stats = defaultdict(lambda: {"total_steps": 0, "correct_steps": 0, "incorrect_steps": 0})
video_to_task = {}
video_to_label = {}

video_idx_counter = 0

for rec_id in hiero_emb.files:
    if rec_id.endswith('_errors'):
        continue # skip the boolean mask arrays

    if rec_id not in boundaries:
        print(f"Warning: {rec_id} not in boundaries!")
        continue
        
    activity_str = boundaries[rec_id]['activity']
    task_name = activity_str.lower().replace(' ', '')
    label = boundaries[rec_id].get('video_label', 0)
    
    vid_idx = video_idx_counter
    video_idx_counter += 1
    
    video_to_task[str(vid_idx)] = task_name
    video_to_label[str(vid_idx)] = label
    
    feats = hiero_emb[rec_id] # (num_steps, 256)
    if feats.ndim == 1:
        feats = feats[np.newaxis, :]
        
    task_to_features[task_name].append(feats)
    
    for step_idx in range(feats.shape[0]):
        task_step_metadata[task_name].append({
            "recording_id": rec_id,
            "video_idx": vid_idx,
            "step_idx_in_video": step_idx,
            "label": label
        })
        task_stats[task_name]["total_steps"] += 1
        if label == 0:
            task_stats[task_name]["correct_steps"] += 1
        else:
            task_stats[task_name]["incorrect_steps"] += 1

reorganized_features = {}
for task_name, feats_list in task_to_features.items():
    reorganized_features[task_name] = np.concatenate(feats_list, axis=0)
    print(f"Task {task_name}: {reorganized_features[task_name].shape[0]} total steps")

npz_out = 'visual_features/reorganized_hiero_steps_all.npz'
np.savez(npz_out, **reorganized_features)
print(f"Saved {npz_out}")

json_out = 'visual_features/hiero_features_mapping.json'
metadata = {
    "video_to_task": video_to_task,
    "video_to_label": video_to_label,
    "task_step_metadata": dict(task_step_metadata),
    "task_stats": dict(task_stats)
}
with open(json_out, 'w') as f:
    json.dump(metadata, f)
print(f"Saved {json_out}")
