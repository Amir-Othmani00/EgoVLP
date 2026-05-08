import json
import argparse
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import pickle
import torch

from train_fusion import FeatureFusionModule


VISUAL_METADATA_CANDIDATES = [
    'gt_features_mapping.json',
    'best_visual_features_mapping.json',
    'hiero_features_mapping.json',
    'hiero_visual_features_mapping.json',
    'visual_features_mapping.json',
    'actionformer_visual_features_mapping.json',
]


def load_embeddings(task_graph_path, visual_features_path, metadata_path):
    """Load task graph and visual embeddings."""
    task_graphs = np.load(task_graph_path)
    visual_features = np.load(visual_features_path)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    task_data = {}
    for task_name in task_graphs.files:
        task_embeddings = task_graphs[task_name]
        visual_emb = visual_features[task_name] if task_name in visual_features.files else None

        task_data[task_name] = {
            'task_embeddings': task_embeddings,
            'visual_embeddings': visual_emb,
            'descriptions': metadata[task_name]['descriptions'],
            'steps': metadata[task_name]['steps'],
            'edges': metadata[task_name]['edges'],
        }

    return task_data, visual_features


def validate_embedding_dimension(task_data, expected_dim):
    for task_name, data in task_data.items():
        task_dim = data['task_embeddings'].shape[1]
        if task_dim != expected_dim:
            raise ValueError(
                f"Task graph dimension mismatch for {task_name}: expected {expected_dim}, got {task_dim}. "
                "Re-encode the task graphs or pass --embedding_dim to match the files you are using."
            )


def resolve_visual_metadata_path(args):
    if args.visual_metadata:
        return Path(args.visual_metadata)

    visual_parent = Path(args.visual_features).parent
    for candidate_name in VISUAL_METADATA_CANDIDATES:
        candidate_path = visual_parent / candidate_name
        if candidate_path.exists():
            return candidate_path

    return visual_parent / 'visual_features_mapping.json'


def load_fusion_model(checkpoint_path, embedding_dim, device):
    if not checkpoint_path:
        return None

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        print(f"⚠ Warning: fusion checkpoint not found at {checkpoint_file}")
        return None

    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    checkpoint_args = checkpoint.get('args', {})

    model = FeatureFusionModule(
        embedding_dim=checkpoint.get('embedding_dim', embedding_dim),
        hidden_dim=checkpoint_args.get('hidden_dim', 512),
        output_dim=checkpoint_args.get('output_dim', embedding_dim),
        fusion_type=checkpoint_args.get('fusion_type', 'concat')
    )

    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded fusion checkpoint from {checkpoint_file}")
    return model


def build_recordings_for_task(task_name, task_data_entry, visual_metadata, visual_features):
    step_metadata_list = visual_metadata.get('task_step_metadata', {}).get(task_name, [])
    recording_groups = {}
    for step_idx, step_meta in enumerate(step_metadata_list):
        recording_id = str(step_meta.get('recording_id', 'unknown'))
        recording_groups.setdefault(recording_id, []).append(step_idx)

    recordings = []

    task_visual_embeddings = task_data_entry.get('visual_embeddings')
    if task_visual_embeddings is not None:
        for recording_id, step_indices in recording_groups.items():
            recordings.append({
                'recording_id': recording_id,
                'visual_embeddings': task_visual_embeddings[step_indices],
                'step_metadata': [step_metadata_list[index] for index in step_indices],
                'visual_indices': step_indices,
            })
        return recordings

    for recording_id, step_indices in recording_groups.items():
        if recording_id not in visual_features.files:
            print(f"Warning: No visual features found for recording {recording_id} in task {task_name}")
            continue

        rec_visual_embeddings = visual_features[recording_id]
        recordings.append({
            'recording_id': recording_id,
            'visual_embeddings': rec_visual_embeddings,
            'step_metadata': [step_metadata_list[index] for index in step_indices],
            'visual_indices': list(range(len(rec_visual_embeddings))),
        })

    return recordings


def fuse_pair(model, task_embedding, visual_embedding, device):
    task_tensor = torch.as_tensor(task_embedding, dtype=torch.float32, device=device).unsqueeze(0)
    visual_tensor = torch.as_tensor(visual_embedding, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        fused_tensor = model(task_tensor, visual_tensor).squeeze(0)

    return fused_tensor.detach().cpu().numpy()


def compute_similarity_matrix(embeddings_a, embeddings_b, metric='cosine'):
    """Compute similarity matrix between two sets of embeddings."""
    if metric == 'cosine':
        # Normalize embeddings
        embeddings_a = embeddings_a / (np.linalg.norm(embeddings_a, axis=1, keepdims=True) + 1e-8)
        embeddings_b = embeddings_b / (np.linalg.norm(embeddings_b, axis=1, keepdims=True) + 1e-8)
        # Cosine similarity
        similarity = np.dot(embeddings_a, embeddings_b.T)
    elif metric == 'euclidean':
        # Negative euclidean distance (for maximization)
        similarity = -np.linalg.norm(embeddings_a[:, None] - embeddings_b[None, :], axis=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return similarity


def hungarian_matching(task_embeddings, visual_embeddings, metric='cosine'):
    """
    Perform Hungarian matching between task graph nodes and visual steps.
    
    Args:
        task_embeddings: (num_nodes, embedding_dim)
        visual_embeddings: (num_steps, embedding_dim)
        metric: 'cosine' or 'euclidean'
    
    Returns:
        matches: list of (task_idx, visual_idx) tuples
        unmatched_task: list of unmatched task indices
        unmatched_visual: list of unmatched visual indices
        similarity_matrix: the similarity matrix
    """
    # Compute similarity matrix
    similarity = compute_similarity_matrix(task_embeddings, visual_embeddings, metric)
    
    # Convert to cost matrix (Hungarian algorithm minimizes cost)
    cost_matrix = -similarity

    task_indices, visual_indices = linear_sum_assignment(cost_matrix)

    matches = list(zip(task_indices, visual_indices))

    all_task_idx = set(range(len(task_embeddings)))
    all_visual_idx = set(range(len(visual_embeddings)))
    matched_task_idx = set(task_indices)
    matched_visual_idx = set(visual_indices)
    
    unmatched_task = list(all_task_idx - matched_task_idx)
    unmatched_visual = list(all_visual_idx - matched_visual_idx)
    
    return matches, unmatched_task, unmatched_visual, similarity


def main(args):
    print("=" * 80)
    print("Loading embeddings...")
    print("=" * 80)
    
    task_data, visual_features = load_embeddings(
        args.task_graph_embeddings,
        args.visual_features,
        args.metadata
    )
    
    print(f"Loaded {len(task_data)} tasks")

    if not task_data:
        raise RuntimeError('No task/visual embeddings were loaded.')

    embedding_dim = args.embedding_dim
    validate_embedding_dimension(task_data, embedding_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fusion_model = load_fusion_model(args.fusion_checkpoint, embedding_dim, device)
    
    print("\nLoading visual features metadata...")
    visual_metadata_path = resolve_visual_metadata_path(args)

    if visual_metadata_path.exists():
        with open(visual_metadata_path, 'r') as f:
            visual_metadata = json.load(f)
    else:
        print(f"⚠ Warning: Visual metadata not found at {visual_metadata_path}")
        visual_metadata = {}
    
    has_step_metadata = 'task_step_metadata' in visual_metadata
    if has_step_metadata:
        print("✓ Step-level metadata found (includes recording IDs)")
    else:
        print("⚠ Warning: No step-level metadata found. Recording IDs will not be included.")
    
    print("\n" + "=" * 80)
    print("Performing Hungarian matching...")
    print("=" * 80)
    
    all_matches = {}
    matched_pairs = []
    updated_recording_embeddings = {}
    recording_metadata = {}
    
    for task_name, data in task_data.items():
        task_emb = data['task_embeddings']
        
        print(f"\nTask: {task_name}")

        recordings = build_recordings_for_task(task_name, data, visual_metadata, visual_features)
        print(f"  Task nodes: {len(task_emb)}, Recordings: {len(recordings)}")

        if not recordings:
            print("  Skipping task because no recording-level visual features were found.")
            continue

        for recording_data in recordings:
            rec_dim = recording_data['visual_embeddings'].shape[1]
            if rec_dim != embedding_dim:
                raise ValueError(
                    f"Visual feature dimension mismatch for task {task_name} recording {recording_data['recording_id']}: "
                    f"expected {embedding_dim}, got {rec_dim}. Pass matching 256-D files or set --embedding_dim explicitly."
                )

        task_level_records = {}
        task_level_unmatched_task = []
        task_level_unmatched_visual = []

        for recording_data in recordings:
            rec_id = recording_data['recording_id']
            rec_visual_emb = recording_data['visual_embeddings']
            rec_step_metadata = recording_data['step_metadata']
            indices = recording_data['visual_indices']

            matches, unmatched_t, unmatched_v, similarity = hungarian_matching(
                task_emb, rec_visual_emb, metric=args.matching_metric
            )

            updated_task_emb = task_emb.copy()
            matched_task_indices = []
            matched_visual_indices = []

            global_matches = []
            for task_idx, local_visual_idx in matches:
                global_visual_idx = indices[local_visual_idx]
                global_matches.append((task_idx, global_visual_idx))
                matched_task_indices.append(int(task_idx))
                matched_visual_indices.append(int(global_visual_idx))

                fused_embedding = task_emb[task_idx]
                if fusion_model is not None:
                    fused_embedding = fuse_pair(
                        fusion_model,
                        task_emb[task_idx],
                        rec_visual_emb[local_visual_idx],
                        device
                    )
                updated_task_emb[task_idx] = fused_embedding

                pair = {
                    'task_embedding': task_emb[task_idx].tolist(),
                    'visual_embedding': rec_visual_emb[local_visual_idx].tolist(),
                    'fused_embedding': fused_embedding.tolist(),
                    'task_name': task_name,
                    'task_idx': int(task_idx),
                    'visual_idx': int(global_visual_idx),
                    'recording_id': rec_id,
                    'similarity': float(similarity[task_idx, local_visual_idx]),
                    'description': data['descriptions'][task_idx] if task_idx < len(data['descriptions']) else 'N/A'
                }

                if local_visual_idx < len(rec_step_metadata):
                    step_meta = rec_step_metadata[local_visual_idx]
                    pair['video_label'] = step_meta.get('label', -1)
                    pair['video_idx'] = step_meta.get('video_idx', -1)
                    pair['step_idx_in_video'] = step_meta.get('step_idx_in_video', -1)

                matched_pairs.append(pair)

            task_level_records[rec_id] = {
                'matches': global_matches,
                'unmatched_task': unmatched_t,
                'unmatched_visual': [indices[v] for v in unmatched_v],
                'recording_id': rec_id,
                'task_name': task_name,
                'num_matches': len(global_matches),
            }
            task_level_unmatched_task.extend(unmatched_t)
            task_level_unmatched_visual.extend([indices[v] for v in unmatched_v])
            updated_recording_embeddings[rec_id] = updated_task_emb
            recording_metadata[rec_id] = {
                'task_name': task_name,
                'label': rec_step_metadata[0].get('label', -1) if rec_step_metadata else -1,
                'matched_task_indices': matched_task_indices,
                'matched_visual_indices': matched_visual_indices,
                'num_steps': len(indices),
            }

        all_matches[task_name] = {
            'recordings': task_level_records,
            'unmatched_task': list(set(task_level_unmatched_task)),
            'unmatched_visual': task_level_unmatched_visual,
        }

        print(f"  Matched pairs: {sum(len(v['matches']) for v in task_level_records.values())}")
        print(f"  Unmatched task nodes: {len(set(task_level_unmatched_task))}")
        print(f"  Unmatched visual steps: {len(task_level_unmatched_visual)}")
    
    print(f"\n" + "=" * 80)
    print(f"Total matched pairs across all tasks: {len(matched_pairs)}")
    print("=" * 80)
    
    # Save matches
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full matches with similarity matrices (pickle for numpy arrays)
    with open(output_dir / 'matches.pkl', 'wb') as f:
        pickle.dump(all_matches, f)
    print(f"\nSaved full matches to {output_dir / 'matches.pkl'}")
    
    # Save matched pairs (JSON for human readability)
    with open(output_dir / 'matched_pairs.json', 'w') as f:
        json.dump(matched_pairs, f, indent=2)
    print(f"Saved matched pairs to {output_dir / 'matched_pairs.json'}")

    updated_npz_path = output_dir / 'updated_task_graph_embeddings.npz'
    np.savez_compressed(
        updated_npz_path,
        **{recording_id: features for recording_id, features in updated_recording_embeddings.items()}
    )
    print(f"Saved updated task graph embeddings to {updated_npz_path}")

    with open(output_dir / 'recording_metadata.json', 'w') as f:
        json.dump(recording_metadata, f, indent=2)
    print(f"Saved recording metadata to {output_dir / 'recording_metadata.json'}")
    
    # Save summary statistics
    summary = {
        'total_tasks': len(task_data),
        'total_matches': len(matched_pairs),
        'matching_metric': args.matching_metric,
        'per_task_stats': {}
    }
    
    for task_name, match_data in all_matches.items():
        num_matches = sum(len(recording_data['matches']) for recording_data in match_data['recordings'].values())
        summary['per_task_stats'][task_name] = {
            'num_matches': num_matches,
            'num_unmatched_task': len(match_data['unmatched_task']),
            'num_unmatched_visual': len(match_data['unmatched_visual'])
        }
    
    with open(output_dir / 'matching_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {output_dir / 'matching_summary.json'}")
    
    print("\n" + "=" * 80)
    print("Matching completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match task graph nodes to visual steps using Hungarian algorithm')
    
    # Input paths
    parser.add_argument('--task_graph_embeddings', type=str,
                        default='outputs/task_graph_encodings_256/task_graph_embeddings.npz',
                        help='Path to task graph embeddings')
    parser.add_argument('--visual_features', type=str,
                        default='visual_features/hiero_step_embeddings_256.npz',
                        help='Path to visual features')
    parser.add_argument('--visual_metadata', type=str,
                        default='visual_features/hiero_visual_features_mapping.json',
                        help='Path to visual features metadata JSON (optional)')
    parser.add_argument('--metadata', type=str,
                        default='outputs/task_graph_encodings_256/task_graph_metadata.json',
                        help='Path to task graph metadata')

    parser.add_argument('--fusion_checkpoint', type=str,
                        default='',
                        help='Optional trained fusion checkpoint to update matched node features')
    
    # Output
    parser.add_argument('--output_dir', type=str,
                        default='outputs/matched_features',
                        help='Directory to save matching results')
    
    # Matching parameters
    parser.add_argument('--matching_metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean'],
                        help='Metric for Hungarian matching')

    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Expected shared embedding dimension for task and visual features')
    
    args = parser.parse_args()
    
    main(args)
