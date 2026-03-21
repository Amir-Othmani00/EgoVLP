import numpy as np
import json
from pathlib import Path
from collections import defaultdict

print("=" * 80)
print("Reorganizing Visual Features with Recording Metadata Preserved")
print("=" * 80)

# Load boundaries with task names and video IDs
print("\nLoading task mapping from hiero_step_boundaries.json...")
with open('visual_features/hiero_step_boundaries.json', 'r') as f:
    boundaries = json.load(f)

# Build comprehensive mapping
video_metadata = {}  # video_idx -> metadata
video_to_task = {}
video_to_label = {}

for idx, (vid_id, info) in enumerate(sorted(boundaries.items())):
    activity = info.get('activity', '').lower().replace(' ', '')
    video_to_task[idx] = activity
    video_to_label[idx] = info.get('video_label', -1)
    video_metadata[idx] = {
        'recording_id': vid_id,
        'task': activity,
        'label': info.get('video_label', -1),
        'boundaries': info.get('boundaries', []),
        'feature_file': info.get('feature_file', '')
    }

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

# Reorganize by task with metadata preservation
print("\n" + "=" * 80)
print("Organizing features by task with metadata...")
print("=" * 80)

# Structure: task -> list of (embeddings, metadata)
task_features = defaultdict(lambda: defaultdict(list))
task_step_metadata = defaultdict(list)  # Store metadata for each step

for video_idx in range(len(step_embeddings)):
    task = video_to_task[video_idx]
    label = video_to_label[video_idx]
    recording_id = video_metadata[video_idx]['recording_id']
    
    # Extract valid steps
    valid_mask = step_mask[video_idx].astype(bool)
    valid_embeddings = step_embeddings[video_idx][valid_mask]
    
    if len(valid_embeddings) > 0:
        # Store embeddings by label
        task_features[task][label].append(valid_embeddings)
        
        # Store metadata for each step
        for step_idx in range(len(valid_embeddings)):
            task_step_metadata[task].append({
                'recording_id': recording_id,
                'video_idx': video_idx,
                'step_idx_in_video': step_idx,
                'label': label
            })

# Aggregate and save
print("\nAggregating features...")

# Create separate NPZ files for different purposes
all_task_embeddings = {}  # All embeddings per task
correct_task_embeddings = {}  # Only correct executions
incorrect_task_embeddings = {}  # Only incorrect executions
task_metadata_dict = {}  # Metadata for each task's steps

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
    
    # Concatenate all embeddings
    if all_embeds_list:
        all_task_embeddings[task_name] = np.vstack(all_embeds_list)
    
    if correct_embeds_list:
        correct_task_embeddings[task_name] = np.vstack(correct_embeds_list)
    
    if incorrect_embeds_list:
        incorrect_task_embeddings[task_name] = np.vstack(incorrect_embeds_list)
    
    # Store metadata
    task_metadata_dict[task_name] = task_step_metadata[task_name]
    
    # Statistics
    num_correct = len(correct_embeds_list)
    num_incorrect = len(incorrect_embeds_list)
    total_videos = num_correct + num_incorrect
    total_steps = all_task_embeddings[task_name].shape[0] if task_name in all_task_embeddings else 0
    
    task_stats[task_name] = {
        'num_correct_videos': num_correct,
        'num_incorrect_videos': num_incorrect,
        'total_videos': total_videos,
        'total_steps': total_steps
    }
    
    print(f"{task_name}:")
    print(f"  Correct: {num_correct} videos, {correct_task_embeddings[task_name].shape[0] if task_name in correct_task_embeddings else 0} steps")
    print(f"  Incorrect: {num_incorrect} videos, {incorrect_task_embeddings[task_name].shape[0] if task_name in incorrect_task_embeddings else 0} steps")
    print(f"  Total: {total_videos} videos, {total_steps} steps")

# Save reorganized features
output_dir = Path('visual_features')
output_dir.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 80)
print("Saving reorganized features...")
print("=" * 80)

# Save all embeddings
np.savez_compressed(
    output_dir / 'reorganized_step_embeddings_all.npz',
    **all_task_embeddings
)
print(f"✓ Saved all embeddings to {output_dir / 'reorganized_step_embeddings_all.npz'}")

# Save correct only
np.savez_compressed(
    output_dir / 'reorganized_step_embeddings_correct.npz',
    **correct_task_embeddings
)
print(f"✓ Saved correct embeddings to {output_dir / 'reorganized_step_embeddings_correct.npz'}")

# Save incorrect only
np.savez_compressed(
    output_dir / 'reorganized_step_embeddings_incorrect.npz',
    **incorrect_task_embeddings
)
print(f"✓ Saved incorrect embeddings to {output_dir / 'reorganized_step_embeddings_incorrect.npz'}")

# Save metadata mapping
metadata_output = {
    'video_to_task': video_to_task,
    'video_to_label': video_to_label,
    'task_step_metadata': task_metadata_dict,  # NEW: step-level metadata
    'task_stats': task_stats,
    'label_meaning': {
        '0': 'correct',
        '1': 'incorrect'
    }
}

with open(output_dir / 'visual_features_mapping.json', 'w') as f:
    json.dump(metadata_output, f, indent=2)
print(f"✓ Saved metadata mapping to {output_dir / 'visual_features_mapping.json'}")

# Save video metadata separately for easy lookup
with open(output_dir / 'video_metadata.json', 'w') as f:
    # Convert int keys to str for JSON
    video_metadata_json = {str(k): v for k, v in video_metadata.items()}
    json.dump(video_metadata_json, f, indent=2)
print(f"✓ Saved video metadata to {output_dir / 'video_metadata.json'}")

print("\n" + "=" * 80)
print("Reorganization complete!")
print("=" * 80)
print(f"\nTotal tasks: {len(all_task_embeddings)}")
print(f"Total steps across all tasks: {sum(emb.shape[0] for emb in all_task_embeddings.values())}")
print(f"\nMetadata preserved:")
print(f"  - Recording IDs")
print(f"  - Video indices")
print(f"  - Step indices within videos")
print(f"  - Correct/incorrect labels")
