import os
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import pickle


def load_embeddings(task_graph_path, visual_features_path, metadata_path):
    """Load task graph and visual embeddings."""
    # Load task graph embeddings
    task_graphs = np.load(task_graph_path)
    
    # Load visual features
    visual_features = np.load(visual_features_path)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Organize data by task
    task_data = {}
    for task_name in task_graphs.files:
        task_embeddings = task_graphs[task_name]  # Shape: (num_nodes, embedding_dim)
        
        # Get corresponding visual features (assuming naming convention)
        visual_key = task_name  # Adjust if visual features have different naming
        if visual_key in visual_features.files:
            visual_emb = visual_features[visual_key]  # Shape: (num_steps, embedding_dim)
        else:
            print(f"Warning: No visual features found for {task_name}")
            continue
        
        task_data[task_name] = {
            'task_embeddings': task_embeddings,
            'visual_embeddings': visual_emb,
            'descriptions': metadata[task_name]['descriptions'],
            'steps': metadata[task_name]['steps'],
            'edges': metadata[task_name]['edges']
        }
    
    return task_data


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
    
    # Apply Hungarian algorithm
    task_indices, visual_indices = linear_sum_assignment(cost_matrix)
    
    # Get matches
    matches = list(zip(task_indices, visual_indices))
    
    # Find unmatched indices
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
    
    task_data = load_embeddings(
        args.task_graph_embeddings,
        args.visual_features,
        args.metadata
    )
    
    print(f"Loaded {len(task_data)} tasks")
    
    print("\n" + "=" * 80)
    print("Performing Hungarian matching...")
    print("=" * 80)
    
    all_matches = {}
    matched_pairs = []
    
    for task_name, data in task_data.items():
        task_emb = data['task_embeddings']
        visual_emb = data['visual_embeddings']
        
        print(f"\nTask: {task_name}")
        print(f"  Task nodes: {len(task_emb)}, Visual steps: {len(visual_emb)}")
        
        matches, unmatched_task, unmatched_visual, similarity = hungarian_matching(
            task_emb, visual_emb, metric=args.matching_metric
        )
        
        print(f"  Matched pairs: {len(matches)}")
        print(f"  Unmatched task nodes: {len(unmatched_task)}")
        print(f"  Unmatched visual steps: {len(unmatched_visual)}")
        
        # Calculate average similarity for matched pairs
        match_similarities = [similarity[t_idx, v_idx] for t_idx, v_idx in matches]
        if match_similarities:
            avg_similarity = np.mean(match_similarities)
            print(f"  Average similarity: {avg_similarity:.4f}")
        
        # Store matches
        all_matches[task_name] = {
            'matches': matches,
            'unmatched_task': unmatched_task,
            'unmatched_visual': unmatched_visual,
            'similarity_matrix': similarity.tolist()  # Convert to list for JSON serialization
        }
        
        # Create matched pairs for downstream training
        for task_idx, visual_idx in matches:
            matched_pairs.append({
                'task_embedding': task_emb[task_idx].tolist(),
                'visual_embedding': visual_emb[visual_idx].tolist(),
                'task_name': task_name,
                'task_idx': int(task_idx),
                'visual_idx': int(visual_idx),
                'similarity': float(similarity[task_idx, visual_idx]),
                'description': data['descriptions'][task_idx] if task_idx < len(data['descriptions']) else 'N/A'
            })
    
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
    
    # Save summary statistics
    summary = {
        'total_tasks': len(task_data),
        'total_matches': len(matched_pairs),
        'matching_metric': args.matching_metric,
        'per_task_stats': {}
    }
    
    for task_name, match_data in all_matches.items():
        summary['per_task_stats'][task_name] = {
            'num_matches': len(match_data['matches']),
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
                        default='outputs/task_graph_encodings/task_graph_embeddings.npz',
                        help='Path to task graph embeddings')
    parser.add_argument('--visual_features', type=str,
                        default='visual_features/hiero_step_embeddings_256.npz',
                        help='Path to visual features')
    parser.add_argument('--metadata', type=str,
                        default='outputs/task_graph_encodings/task_graph_metadata.json',
                        help='Path to task graph metadata')
    
    # Output
    parser.add_argument('--output_dir', type=str,
                        default='outputs/matched_features',
                        help='Directory to save matching results')
    
    # Matching parameters
    parser.add_argument('--matching_metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean'],
                        help='Metric for Hungarian matching')
    
    args = parser.parse_args()
    
    main(args)
