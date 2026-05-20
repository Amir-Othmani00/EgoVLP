import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_ANNOTATIONS = "annotations/annotation_json/complete_step_annotations.json"
DEFAULT_HIERO_STEP_METADATA = "visual_features/hiero_results.json"
DEFAULT_HIERO_EMBEDDINGS = "visual_features/hiero_embeddings.npz"
DEFAULT_HIERO_OUT_NPZ = "visual_features/hiero_step_embeddings_256.npz"
DEFAULT_HIERO_OUT_JSON = "visual_features/hiero_visual_features_mapping.json"

DEFAULT_GT_STEPS = "visual_features/gt-steps.npz"
DEFAULT_GT_OUT_NPZ = "visual_features/gt_steps_reorganized.npz"
DEFAULT_GT_OUT_JSON = "visual_features/gt_features_mapping.json"

DEFAULT_ACTIONFORMER_METADATA = "visual_features/actionformer_steps.json"
DEFAULT_ACTIONFORMER_EMBEDDINGS = "visual_features/actionformer_steps.npz"
DEFAULT_ACTIONFORMER_OUT_NPZ = "visual_features/actionformer_step_embeddings_256.npz"
DEFAULT_ACTIONFORMER_OUT_JSON = "visual_features/actionformer_visual_features_mapping.json"


def normalize_task_name(activity_name):
    return str(activity_name).lower().replace(" ", "")


def has_error_step(step_entries):
    return 1 if any(step.get("has_errors", False) for step in step_entries) else 0


def load_annotations(path):
    with open(path, "r") as file_handle:
        return json.load(file_handle)


def aggregate_hiero_steps(annotations, step_metadata, embeddings, strip_errors=False):
    task_to_features = defaultdict(list)
    task_step_metadata = defaultdict(list)
    task_stats = defaultdict(lambda: {"total_steps": 0, "correct_steps": 0, "incorrect_steps": 0})
    video_to_task = {}
    video_to_label = {}

    video_idx_counter = 0

    for recording_id in embeddings.files:
        if recording_id not in annotations:
            print(f"Warning: {recording_id} not in annotations!")
            continue

        activity_name = annotations[recording_id].get("activity_name", "")
        task_name = normalize_task_name(activity_name)
        label = has_error_step(annotations[recording_id].get("steps", []))

        video_idx = video_idx_counter
        video_idx_counter += 1

        video_to_task[str(video_idx)] = task_name
        video_to_label[str(video_idx)] = label

        features = embeddings[recording_id]
        if features.ndim == 1:
            features = features[np.newaxis, :]

        task_to_features[task_name].append(features)

        step_info_list = step_metadata.get(recording_id, {}).get("steps", [])

        for step_idx in range(features.shape[0]):
            step_meta = {
                "recording_id": recording_id,
                "video_idx": video_idx,
                "step_idx_in_video": step_idx,
            }

            step_meta["label"] = label

            if strip_errors:
                # Keep the task-level label, but remove explicit error flags.
                step_meta.pop("has_errors", None)

            if step_idx < len(step_info_list):
                step_meta.update(step_info_list[step_idx])

            task_step_metadata[task_name].append(step_meta)

            task_stats[task_name]["total_steps"] += 1
            if label == 0:
                task_stats[task_name]["correct_steps"] += 1
            else:
                task_stats[task_name]["incorrect_steps"] += 1

    reorganized_features = {
        task_name: np.concatenate(feature_list, axis=0)
        for task_name, feature_list in task_to_features.items()
    }

    metadata = {
        "video_to_task": video_to_task,
        "video_to_label": video_to_label,
        "task_step_metadata": dict(task_step_metadata),
        "task_stats": dict(task_stats),
    }

    return reorganized_features, metadata


def aggregate_gt_steps(annotations, gt_data, strip_errors=False):
    recording_to_task = {recording_id: normalize_task_name(info.get("activity_name", ""))
                         for recording_id, info in annotations.items()}
    recording_to_label = {recording_id: has_error_step(info.get("steps", []))
                          for recording_id, info in annotations.items()}

    files_list = list(gt_data.files)

    # Two possible GT formats:
    # 1) Keys like 'step_{i}_features' / 'step_{i}_recording_id' (older format)
    # 2) Per-recording keys where each key is a recording_id and value is (num_steps, dim)
    if any(name.startswith("step_") for name in files_list):
        # existing behavior
        step_indices = sorted({name.split("_")[1] for name in files_list if name.startswith("step_")}, key=int)
        gt_dict = {key: gt_data[key] for key in files_list}
        task_features = defaultdict(list)
        task_step_metadata = defaultdict(list)

        for step_idx in step_indices:
            recording_key = f"step_{step_idx}_recording_id"
            feature_key = f"step_{step_idx}_features"
            description_key = f"step_{step_idx}_description"
            error_key = f"step_{step_idx}_has_errors"
            start_key = f"step_{step_idx}_start_time"
            end_key = f"step_{step_idx}_end_time"

            if recording_key not in files_list or feature_key not in files_list:
                continue

            recording_id = str(gt_dict[recording_key])
            features = gt_dict[feature_key]
            step_embedding = np.mean(features, axis=0) if len(features) > 0 else np.zeros(features.shape[1])

            task_name = recording_to_task.get(recording_id)
            if not task_name:
                continue

            task_features[task_name].append(step_embedding)
            
            # Try to find step_id from annotations if description matches
            desc = str(gt_dict[description_key]) if description_key in files_list else ""
            ann_steps = annotations.get(recording_id, {}).get('steps', [])
            step_id = -1
            for ann_step in ann_steps:
                if ann_step.get('description', '') == desc:
                    step_id = ann_step.get('step_id', -1)
                    break

            entry = {
                "recording_id": recording_id,
                "step_idx_in_video": int(step_idx),
                "step_id": int(step_id),
                "label": recording_to_label.get(recording_id, -1),
                "description": desc,
                "start_time": float(gt_dict[start_key]) if start_key in files_list else -1.0,
                "end_time": float(gt_dict[end_key]) if end_key in files_list else -1.0,
            }
            if not strip_errors:
                entry["has_errors"] = bool(gt_dict[error_key]) if error_key in files_list else False

            task_step_metadata[task_name].append(entry)

        all_task_embeddings = {task_name: np.vstack(step_embeddings) for task_name, step_embeddings in task_features.items()}
        metadata = {"task_step_metadata": dict(task_step_metadata)}
        return all_task_embeddings, metadata

    # Handle per-recording GT format: each key is a recording_id and value is (num_steps, dim)
    task_features = defaultdict(list)
    task_step_metadata = defaultdict(list)

    for recording_key in files_list:
        # skip any helper keys
        if recording_key.startswith("used_"):
            continue

        recording_id = str(recording_key)
        if recording_id not in recording_to_task:
            # skip recordings not present in annotations
            continue

        features = gt_data[recording_key]
        # each row corresponds to a step embedding
        num_steps = features.shape[0]
        ann_steps = annotations.get(recording_id, {}).get('steps', [])
        
        if len(ann_steps) != num_steps:
            print(f"Warning: Recording {recording_id} has {num_steps} steps in GT features but {len(ann_steps)} in annotations.")

        for step_idx in range(num_steps):
            step_embedding = features[step_idx]
            task_name = recording_to_task.get(recording_id)
            if not task_name:
                continue

            task_features[task_name].append(step_embedding)
            
            # Pull metadata from annotations
            if step_idx < len(ann_steps):
                step_info = ann_steps[step_idx]
                step_id = step_info.get('step_id', -1)
                desc = step_info.get('description', '')
                start_time = step_info.get('start_time', -1.0)
                end_time = step_info.get('end_time', -1.0)
                has_err = step_info.get('has_errors', False)
            else:
                step_id = -1
                desc = ""
                start_time = -1.0
                end_time = -1.0
                has_err = False

            entry = {
                "recording_id": recording_id,
                "step_idx_in_video": int(step_idx),
                "step_id": int(step_id),
                "description": str(desc),
                "start_time": float(start_time),
                "end_time": float(end_time),
            }
            if not strip_errors:
                entry["label"] = recording_to_label.get(recording_id, -1)
                entry["has_errors"] = bool(has_err)

            task_step_metadata[task_name].append(entry)

    all_task_embeddings = {task_name: np.vstack(step_embeddings) for task_name, step_embeddings in task_features.items() if len(step_embeddings)>0}
    metadata = {"task_step_metadata": dict(task_step_metadata)}

    return all_task_embeddings, metadata


def aggregate_actionformer_steps(annotations, step_metadata, embeddings, strip_errors=False):
    recording_to_task = {recording_id: normalize_task_name(info.get("activity_name", ""))
                         for recording_id, info in annotations.items()}
    recording_to_label = {recording_id: has_error_step(info.get("steps", []))
                          for recording_id, info in annotations.items()}

    task_features = defaultdict(list)
    task_step_metadata = defaultdict(list)

    files_list = list(embeddings.files)

    for recording_key in files_list:
        recording_id = str(recording_key)

        if recording_id not in recording_to_task:
            # skip recordings not present in annotations
            continue

        features = embeddings[recording_key]
        # each row corresponds to a step embedding
        num_steps = features.shape[0]

        # Get step metadata from JSON
        step_info_list = step_metadata.get(recording_id, {}).get("steps", [])

        for step_idx in range(num_steps):
            step_embedding = features[step_idx]
            task_name = recording_to_task.get(recording_id)
            if not task_name:
                continue

            task_features[task_name].append(step_embedding)

            # Build metadata entry for this step
            entry = {
                "recording_id": recording_id,
                "step_idx_in_video": int(step_idx),
            }
            entry["label"] = recording_to_label.get(recording_id, -1)

            # Add step-specific information from actionformer metadata
            if step_idx < len(step_info_list):
                step_info = step_info_list[step_idx]
                entry["step_id"] = step_info.get("step_id", -1)
                entry["start_time"] = float(step_info.get("start_time", -1.0))
                entry["end_time"] = float(step_info.get("end_time", -1.0))
            else:
                entry["step_id"] = -1
                entry["start_time"] = -1.0
                entry["end_time"] = -1.0

            if not strip_errors:
                # Try to pull has_errors from annotations if available
                ann_steps = annotations.get(recording_id, {}).get('steps', [])
                has_err = ann_steps[step_idx].get('has_errors', False) if step_idx < len(ann_steps) else False
                entry["has_errors"] = bool(has_err)

            task_step_metadata[task_name].append(entry)

    all_task_embeddings = {task_name: np.vstack(step_embeddings) for task_name, step_embeddings in task_features.items() if len(step_embeddings) > 0}
    metadata = {"task_step_metadata": dict(task_step_metadata)}

    return all_task_embeddings, metadata


def save_outputs(embeddings_by_task, metadata, out_npz, out_json, compressed=False):
    output_path = Path(out_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if compressed:
        np.savez_compressed(output_path, **embeddings_by_task)
    else:
        np.savez(output_path, **embeddings_by_task)

    with open(out_json, "w") as file_handle:
        json.dump(metadata, file_handle, indent=2)


def build_parser():
    parser = argparse.ArgumentParser(description="Reorganize task-step features from different input formats.")
    parser.add_argument("--source", choices=["hiero", "gt", "actionformer"], default="hiero", help="Input format to reorganize")
    parser.add_argument("--annotations", type=str, default=None, help="Path to complete annotations json")
    parser.add_argument("--step_metadata", type=str, default=None, help="Path to step metadata json (hiero/actionformer mode)")
    parser.add_argument("--embeddings", type=str, default=None, help="Path to feature embeddings npz (hiero/actionformer mode)")
    parser.add_argument("--gt_steps", type=str, default=None, help="Path to gt_steps.npz (gt mode)")
    parser.add_argument("--out_npz", type=str, default=None, help="Output npz path")
    parser.add_argument("--out_json", type=str, default=None, help="Output json path")
    parser.add_argument('--strip_errors', action='store_true', help='Omit explicit has_errors fields from produced metadata')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    annotations_path = args.annotations or DEFAULT_ANNOTATIONS
    annotations = load_annotations(annotations_path)

    if args.source == "hiero":
        step_metadata_path = args.step_metadata or DEFAULT_HIERO_STEP_METADATA
        embeddings_path = args.embeddings or DEFAULT_HIERO_EMBEDDINGS
        out_npz = args.out_npz or DEFAULT_HIERO_OUT_NPZ
        out_json = args.out_json or DEFAULT_HIERO_OUT_JSON

        print(f"Loading annotations for task mappings from {annotations_path}...")
        print(f"Loading step metadata from {step_metadata_path}...")
        with open(step_metadata_path, "r") as file_handle:
            step_metadata = json.load(file_handle)

        print(f"Loading feature embeddings from {embeddings_path}...")
        embeddings = np.load(embeddings_path)

        task_embeddings, metadata = aggregate_hiero_steps(annotations, step_metadata, embeddings)

        for task_name, task_features in task_embeddings.items():
            print(f"Task {task_name}: {task_features.shape[0]} total steps")

        save_outputs(task_embeddings, metadata, out_npz, out_json, compressed=False)
        print(f"Saved {out_npz}")
        print(f"Saved {out_json}")
        print("Done grouping embeddings by task.")
        return

    if args.source == "actionformer":
        step_metadata_path = args.step_metadata or DEFAULT_ACTIONFORMER_METADATA
        embeddings_path = args.embeddings or DEFAULT_ACTIONFORMER_EMBEDDINGS
        out_npz = args.out_npz or DEFAULT_ACTIONFORMER_OUT_NPZ
        out_json = args.out_json or DEFAULT_ACTIONFORMER_OUT_JSON

        print(f"Loading annotations for task mappings from {annotations_path}...")
        print(f"Loading step metadata from {step_metadata_path}...")
        with open(step_metadata_path, "r") as file_handle:
            step_metadata = json.load(file_handle)

        print(f"Loading feature embeddings from {embeddings_path}...")
        embeddings = np.load(embeddings_path)

        task_embeddings, metadata = aggregate_actionformer_steps(annotations, step_metadata, embeddings, strip_errors=args.strip_errors)

        for task_name, task_features in task_embeddings.items():
            print(f"Task {task_name}: {task_features.shape[0]} total steps")

        save_outputs(task_embeddings, metadata, out_npz, out_json, compressed=True)
        print(f"Saved {out_npz}")
        print(f"Saved {out_json}")
        print("Done grouping actionformer embeddings by task.")
        return

    gt_steps_path = args.gt_steps or DEFAULT_GT_STEPS
    out_npz = args.out_npz or DEFAULT_GT_OUT_NPZ
    out_json = args.out_json or DEFAULT_GT_OUT_JSON

    print("=" * 80)
    print("Reorganizing GT Steps Features")
    print("=" * 80)
    print(f"\nLoading {gt_steps_path}...")

    gt_data = np.load(gt_steps_path, allow_pickle=True)
    task_embeddings, metadata = aggregate_gt_steps(annotations, gt_data, strip_errors=args.strip_errors if hasattr(args, 'strip_errors') else False)
    gt_data.close()

    save_outputs(task_embeddings, metadata, out_npz, out_json, compressed=True)

    print(f"\nDone! Prepared {Path(out_npz).name} and {Path(out_json).name}")


if __name__ == "__main__":
    main()