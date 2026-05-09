
import json
import numpy as np

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def check_fusion_quality(matched_pairs_path):
    with open(matched_pairs_path, 'r') as f:
        matched_pairs = json.load(f)
    
    if not matched_pairs:
        print("No matched pairs found.")
        return

    # Check if fused_embedding exists in the first pair
    if 'fused_embedding' not in matched_pairs[0]:
        print("Error: 'fused_embedding' not found in matched pairs. Make sure you ran match_features.py with --fusion_checkpoint.")
        return

    orig_sims = []
    fused_task_sims = []
    fused_vis_sims = []
    
    for pair in matched_pairs:
        task_emb = pair['task_embedding']
        vis_emb = pair['visual_embedding']
        fused_emb = pair['fused_embedding']
        
        orig_sims.append(cosine_similarity(task_emb, vis_emb))
        fused_task_sims.append(cosine_similarity(fused_emb, task_emb))
        fused_vis_sims.append(cosine_similarity(fused_emb, vis_emb))
    
    print("=" * 50)
    print("FUSION QUALITY REPORT")
    print("=" * 50)
    print(f"Number of pairs analyzed: {len(matched_pairs)}")
    print("-" * 50)
    print(f"Average Original Similarity (Task <-> Visual): {np.mean(orig_sims):.4f}")
    print(f"Average Fused Similarity (Fused <-> Task):     {np.mean(fused_task_sims):.4f}")
    print(f"Average Fused Similarity (Fused <-> Visual):   {np.mean(fused_vis_sims):.4f}")
    print("-" * 50)
    
    # Improvement check
    improvement = np.mean(fused_task_sims) + np.mean(fused_vis_sims) - (2 * np.mean(orig_sims))
    print(f"Total Alignment Improvement: {improvement:.4f}")
    
    if np.mean(fused_task_sims) > 0.8 and np.mean(fused_vis_sims) > 0.8:
        print("\nStatus: SUCCESS - The fusion model has successfully aligned the features.")
    elif np.mean(fused_task_sims) > np.mean(orig_sims):
        print("\nStatus: PARTIAL - Some alignment learned, but similarities are still low.")
    else:
        print("\nStatus: FAILURE - No improvement in alignment.")
    print("=" * 50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--matched_pairs', type=str, required=True)
    args = parser.parse_args()
    check_fusion_quality(args.matched_pairs)
