import numpy as np
import json
from pathlib import Path
from collections import defaultdict

print("=" * 80)
print("Reorganizing Visual Features with Correct Task Mapping")
print("=" * 80)

# Load boundaries with task names
print("\nLoading task mapping from hiero_step_boundaries.json...")
with open('visual_features/hiero_step_boundaries.json', 'r') as f:
    boundaries = json.load(f)

# Build mapping from video index to task name
video_to_task = {}
video_to_label = {}  # 0=correct, 1=incorrect

for idx, (vid_id, info) in enumerate(sorted(boundaries.items())):
    activity = info.get('activity', '').lower().replace(' ', '')
    video_to_task[idx] = activity
    video_to_label[idx] = info.get('video_label', -1)

print(f"Mapped {len(video_to_task)} videos to tasks")

# Get unique tasks
unique_tasks = set(video_to_task.values())
print(f"Unique tasks found: {len(unique_tasks)}")
print(f"Tasks: {sorted(unique_tasks)}")

# Load visual features
print("\nLoading visual features...")
vf = np.load('visual_features/hiero_step_embeddings_256.npz')
step_embeddings = vf['step_embeddings']  # (384, 61, 256)
step_mask = vf['step_mask']              # (384, 61)

print(f"Step embeddings shape: {step_embeddings.shape}")
print(f"Step mask shape: {step_mask.shape}")

# Load task graph to verify alignment
print("\nLoading task graphs for verification...")
tg = np.load('outputs/task_graph_encodings/task_graph_embeddings.npz')
tg_tasks = set(tg.files)

print(f"Task graph tasks: {len(tg_tasks)}")
print(f"Visual feature tasks: {len(unique_tasks)}")

# Check alignment
common_tasks = unique_tasks & tg_tasks
print(f"Common tasks: {len(common_tasks)}")

if common_tasks:
    print(f"  {sorted(common_tasks)}")

missing_in_visual = tg_tasks - unique_tasks
if missing_in_visual:
    print(f"\nTasks in task graphs but not in visual features:")
    print(f"  {sorted(missing_in_visual)}")

# Reorganize by task and correctness
print("\n" + "=" * 80)
print("Organizing features by task and correctness...")
print("=" * 80)

# Structure: task -> label -> embeddings
task_features = defaultdict(lambda: defaultdict(list))

for video_idx in range(len(step_embeddings)):
    task = video_to_task[video_idx]
    label = video_to_label[video_idx]
    
    # Extract valid steps
    valid_mask = step_mask[video_idx].astype(bool)
    valid_embeddings = step_embeddings[video_idx][valid_mask]
    
    if len(valid_embeddings) > 0:
        task_features[task][label].append(valid_embeddings)

# Aggregate and save
print("\nAggregating features...")

# Create separate NPZ files for different purposes
all_task_embeddings = {}  # All embeddings per task
correct_task_embeddings = {}  # Only correct executions
incorrect_task_embeddings = {}  # Only incorrect executions

task_stats = {}

for task_name in sorted(task_features.keys()):
    # Collect all embeddings for this task
    all_embeds_list = []
    correct_embeds_list = []
    incorrect_embeds_list = []
    
    # Correct embeddings (label 0)
    if 0 in task_features[task_name]:
        for emb_array in task_features[task_name][0]:
            correct_embeds_list.append(emb_array)
            all_embeds_list.append(emb_array)
    
    # Incorrect embeddings (label 1)
    if 1 in task_features[task_name]:
        for emb_array in task_features[task_name][1]:
            incorrect_embeds_list.append(emb_array)
            all_embeds_list.append(emb_array)
    
    # Concatenate properly
    if all_embeds_list:
        all_task_embeddings[task_name] = np.concatenate(all_embeds_list, axis=0)
    
    if correct_embeds_list:
        correct_task_embeddings[task_name] = np.concatenate(correct_embeds_list, axis=0)
    
    if incorrect_embeds_list:
        incorrect_task_embeddings[task_name] = np.concatenate(incorrect_embeds_list, axis=0)
    
    # Stats
    task_stats[task_name] = {
        'total_steps': len(all_task_embeddings[task_name]) if task_name in all_task_embeddings else 0,
        'correct_steps': len(correct_task_embeddings[task_name]) if task_name in correct_task_embeddings else 0,
        'incorrect_steps': len(incorrect_task_embeddings[task_name]) if task_name in incorrect_task_embeddings else 0,
        'num_correct_videos': len(task_features[task_name].get(0, [])),
        'num_incorrect_videos': len(task_features[task_name].get(1, [])),
    }
    
    print(f"\n{task_name}:")
    print(f"  Total steps: {task_stats[task_name]['total_steps']}")
    print(f"  Correct videos: {task_stats[task_name]['num_correct_videos']}, steps: {task_stats[task_name]['correct_steps']}")
    print(f"  Incorrect videos: {task_stats[task_name]['num_incorrect_videos']}, steps: {task_stats[task_name]['incorrect_steps']}")

# Save reorganized features
output_dir = Path('visual_features')

print("\n" + "=" * 80)
print("Saving reorganized features...")
print("=" * 80)

# All features combined
np.savez_compressed(
    output_dir / 'reorganized_step_embeddings_all.npz',
    **all_task_embeddings
)
print(f"Saved all features to reorganized_step_embeddings_all.npz")

# Correct only
np.savez_compressed(
    output_dir / 'reorganized_step_embeddings_correct.npz',
    **correct_task_embeddings
)
print(f"Saved correct only to reorganized_step_embeddings_correct.npz")

# Incorrect only
np.savez_compressed(
    output_dir / 'reorganized_step_embeddings_incorrect.npz',
    **incorrect_task_embeddings
)
print(f"Saved incorrect only to reorganized_step_embeddings_incorrect.npz")

# Save mapping and stats
mapping = {
    'video_to_task': video_to_task,
    'video_to_label': video_to_label,
    'label_meaning': {0: 'correct', 1: 'incorrect'},
    'task_stats': task_stats
}

with open(output_dir / 'visual_features_mapping.json', 'w') as f:
    json.dump(mapping, f, indent=2)
print(f"Saved mapping to visual_features_mapping.json")

# Verify alignment with task graphs
print("\n" + "=" * 80)
print("Verification with Task Graphs:")
print("=" * 80)

print(f"\nAligned tasks ({len(common_tasks)}):")
for task in sorted(common_tasks):
    tg_shape = tg[task].shape
    vf_shape = all_task_embeddings[task].shape
    print(f"  {task}:")
    print(f"    Task graph nodes: {tg_shape[0]}")
    print(f"    Visual steps: {vf_shape[0]}")

if missing_in_visual:
    print(f"\nMissing in visual features ({len(missing_in_visual)}):")
    for task in sorted(missing_in_visual):
        print(f"  {task}")

print("\n" + "=" * 80)
print("Done!")
print("=" * 80)
print("\nYou can now use these files for matching:")
print("  - reorganized_step_embeddings_all.npz (all videos)")
print("  - reorganized_step_embeddings_correct.npz (correct only)")
print("  - reorganized_step_embeddings_incorrect.npz (incorrect only)")
