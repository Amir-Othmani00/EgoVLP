import numpy as np
import json
from pathlib import Path
from collections import defaultdict

print("=" * 80)
print("Reorganizing GT Steps Features")
print("=" * 80)

# 1. Load the task mapping
with open('visual_features/hiero_step_boundaries.json', 'r') as f:
    boundaries = json.load(f)

recording_to_task = {}
recording_to_label = {}
for vid_id, info in boundaries.items():
    recording_to_task[vid_id] = info.get('activity', '').lower().replace(' ', '')
    recording_to_label[vid_id] = info.get('video_label', -1)

# 2. Load gt_steps.npz efficiently
print("\nLoading gt_steps.npz...")
gt_data = np.load('visual_features/gt_steps.npz', allow_pickle=True)
files_list = set(gt_data.files)

step_indices = set(k.split('_')[1] for k in files_list if k.startswith('step_'))
print(f"Found {len(step_indices)} steps in gt_steps.npz")

print("Caching data into dictionary...")
gt_dict = {k: gt_data[k] for k in files_list}
gt_data.close()
print("Dictionary loaded.")

# 3. Aggregate into tasks
task_features = defaultdict(list)
task_step_metadata = defaultdict(list)

for idx in sorted(step_indices):
    rec_key, feat_key = f'step_{idx}_recording_id', f'step_{idx}_features'
    desc_key, err_key = f'step_{idx}_description', f'step_{idx}_has_errors'
    start_key, end_key = f'step_{idx}_start_time', f'step_{idx}_end_time'
    
    if rec_key not in files_list or feat_key not in files_list: continue
        
    rec_id = str(gt_dict[rec_key])
    features = gt_dict[feat_key]
    
    # Mean-pool across frames to get a single (256,) embedding per step
    step_emb = np.mean(features, axis=0) if len(features) > 0 else np.zeros(256)
        
    task = recording_to_task.get(rec_id)
    if not task: continue
    
    task_features[task].append(step_emb)
    task_step_metadata[task].append({
        'recording_id': rec_id,
        'step_idx_in_video': int(idx),
        'label': recording_to_label.get(rec_id, -1),
        'description': str(gt_dict[desc_key]) if desc_key in files_list else '',
        'has_errors': bool(gt_dict[err_key]) if err_key in files_list else False,
        'start_time': float(gt_dict[start_key]) if start_key in files_list else -1.0,
        'end_time': float(gt_dict[end_key]) if end_key in files_list else -1.0
    })

# 4. Save to expected formats
output_dir = Path('visual_features')
output_dir.mkdir(parents=True, exist_ok=True)
all_task_embeddings = {task: np.vstack(embeds) for task, embeds in task_features.items()}

np.savez_compressed(output_dir / 'reorganized_gt_steps_all.npz', **all_task_embeddings)
with open(output_dir / 'gt_features_mapping.json', 'w') as f:
    json.dump({'task_step_metadata': dict(task_step_metadata)}, f, indent=2)

print("\nDone! Prepared reorganized_gt_steps_all.npz and gt_features_mapping.json")