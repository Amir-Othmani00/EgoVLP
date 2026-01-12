import numpy as np
import json
from pathlib import Path
from collections import defaultdict


def create_label_to_task_mapping(task_graph_metadata_path):
    """
    Create a mapping from numeric labels to task names.
    Assumes labels are assigned alphabetically to task names.
    """
    with open(task_graph_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get task names and sort them alphabetically
    task_names = sorted(metadata.keys())
    
    # Create label to task name mapping
    label_to_task = {i: task_name for i, task_name in enumerate(task_names)}
    
    return label_to_task, task_names


def reorganize_visual_features(visual_features_path, label_to_task, output_path):
    """
    Reorganize visual features from batched format to per-task format.
    
    Args:
        visual_features_path: Path to batched visual features
        label_to_task: Dict mapping label indices to task names
        output_path: Path to save reorganized features
    """
    print("Loading visual features...")
    vf = np.load(visual_features_path)
    
    step_embeddings = vf['step_embeddings']  # Shape: (num_videos, max_steps, embedding_dim)
    step_mask = vf['step_mask']              # Shape: (num_videos, max_steps)
    labels = vf['labels']                     # Shape: (num_videos,)
    
    print(f"Loaded {len(labels)} videos")
    print(f"Step embeddings shape: {step_embeddings.shape}")
    
    # Organize by task
    task_features = defaultdict(list)
    task_video_counts = defaultdict(int)
    
    for video_idx, label in enumerate(labels):
        task_name = label_to_task.get(int(label))
        
        if task_name is None:
            print(f"Warning: Unknown label {label} for video {video_idx}")
            continue
        
        # Get valid steps (where mask is True/1)
        valid_steps_mask = step_mask[video_idx].astype(bool)
        valid_embeddings = step_embeddings[video_idx][valid_steps_mask]
        
        if len(valid_embeddings) > 0:
            task_features[task_name].append(valid_embeddings)
            task_video_counts[task_name] += 1
    
    print(f"\nFound videos for {len(task_features)} tasks")
    
    # Aggregate embeddings per task
    task_embeddings = {}
    for task_name, embeddings_list in task_features.items():
        # Concatenate all steps from all videos for this task
        all_steps = np.concatenate(embeddings_list, axis=0)
        task_embeddings[task_name] = all_steps
        
        print(f"  {task_name}: {task_video_counts[task_name]} videos, {len(all_steps)} total steps")
    
    # Save in npz format matching task graph structure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(output_path, **task_embeddings)
    
    print(f"\nSaved reorganized visual features to {output_path}")
    
    return task_embeddings


def verify_alignment(task_graph_path, reorganized_visual_path):
    """Verify that task names align between both files."""
    print("\n" + "=" * 80)
    print("Verifying alignment...")
    print("=" * 80)
    
    tg = np.load(task_graph_path)
    vf = np.load(reorganized_visual_path)
    
    tg_tasks = set(tg.files)
    vf_tasks = set(vf.files)
    
    common_tasks = tg_tasks & vf_tasks
    only_in_tg = tg_tasks - vf_tasks
    only_in_vf = vf_tasks - tg_tasks
    
    print(f"\nTask graph tasks: {len(tg_tasks)}")
    print(f"Visual feature tasks: {len(vf_tasks)}")
    print(f"Common tasks: {len(common_tasks)}")
    
    if only_in_tg:
        print(f"\nTasks only in task graph: {sorted(only_in_tg)}")
    
    if only_in_vf:
        print(f"\nTasks only in visual features: {sorted(only_in_vf)}")
    
    if common_tasks:
        print(f"\nCommon tasks: {sorted(common_tasks)}")
        
        # Show some statistics for common tasks
        print("\nShape comparison for common tasks:")
        for task in sorted(list(common_tasks))[:5]:  # Show first 5
            tg_shape = tg[task].shape
            vf_shape = vf[task].shape
            print(f"  {task}:")
            print(f"    Task graph nodes: {tg_shape[0]}, Visual steps: {vf_shape[0]}")
    
    return common_tasks


def main():
    print("=" * 80)
    print("Reorganizing Visual Features")
    print("=" * 80)
    
    # Paths
    task_graph_embeddings = 'outputs/task_graph_encodings/task_graph_embeddings.npz'
    task_graph_metadata = 'outputs/task_graph_encodings/task_graph_metadata.json'
    visual_features_original = 'visual_features/hiero_step_embeddings_256.npz'
    visual_features_reorganized = 'visual_features/reorganized_step_embeddings.npz'
    
    # Create label to task mapping
    print("\nCreating label to task name mapping...")
    label_to_task, task_names = create_label_to_task_mapping(task_graph_metadata)
    
    print(f"Found {len(task_names)} tasks")
    print("Label to task mapping:")
    for label, task in sorted(label_to_task.items())[:10]:  # Show first 10
        print(f"  {label}: {task}")
    if len(label_to_task) > 10:
        print(f"  ... and {len(label_to_task) - 10} more")
    
    # Save mapping for reference
    mapping_path = Path('visual_features/label_to_task_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(label_to_task, f, indent=2)
    print(f"\nSaved mapping to {mapping_path}")
    
    # Reorganize visual features
    print("\n" + "=" * 80)
    print("Reorganizing visual features by task...")
    print("=" * 80)
    
    task_embeddings = reorganize_visual_features(
        visual_features_original,
        label_to_task,
        visual_features_reorganized
    )
    
    # Verify alignment
    verify_alignment(task_graph_embeddings, visual_features_reorganized)
    
    print("\n" + "=" * 80)
    print("Done! You can now run:")
    print("  python match_features.py --visual_features visual_features/reorganized_step_embeddings.npz")
    print("=" * 80)


if __name__ == '__main__':
    main()
