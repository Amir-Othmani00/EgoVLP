import os
import json
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from train_fusion import FeatureFusionModule, MatchedPairsDataset, contrastive_loss

def normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

def evaluate_retrieval(task_embs, visual_embs):
    # Normalize features for cosine similarity
    task_embs = normalize(task_embs)
    visual_embs = normalize(visual_embs)
    
    # Compute similarity matrix (Cosine similarity since L2 normalized)
    sim_matrix = np.dot(task_embs, visual_embs.T)
    
    n = len(task_embs)
    
    # Text to Video (Task to Visual)
    t2v_ranks = []
    for i in range(n):
        # Sort indices by similarity in descending order
        sorted_indices = np.argsort(-sim_matrix[i])
        rank = np.where(sorted_indices == i)[0][0]
        t2v_ranks.append(rank)
        
    t2v_ranks = np.array(t2v_ranks)
    t2v_r1 = 100.0 * np.mean(t2v_ranks < 1)
    t2v_r5 = 100.0 * np.mean(t2v_ranks < 5)
    t2v_r10 = 100.0 * np.mean(t2v_ranks < 10)
    t2v_medR = np.median(t2v_ranks) + 1
    t2v_meanR = np.mean(t2v_ranks) + 1
    
    # Video to Text (Visual to Task)
    v2t_ranks = []
    for i in range(n):
        sorted_indices = np.argsort(-sim_matrix[:, i])
        rank = np.where(sorted_indices == i)[0][0]
        v2t_ranks.append(rank)
        
    v2t_ranks = np.array(v2t_ranks)
    v2t_r1 = 100.0 * np.mean(v2t_ranks < 1)
    v2t_r5 = 100.0 * np.mean(v2t_ranks < 5)
    v2t_r10 = 100.0 * np.mean(v2t_ranks < 10)
    v2t_medR = np.median(v2t_ranks) + 1
    v2t_meanR = np.mean(v2t_ranks) + 1
    
    metrics = {
        't2v_r1': t2v_r1, 't2v_r5': t2v_r5, 't2v_r10': t2v_r10, 
        't2v_medR': t2v_medR, 't2v_meanR': t2v_meanR,
        'v2t_r1': v2t_r1, 'v2t_r5': v2t_r5, 'v2t_r10': v2t_r10, 
        'v2t_medR': v2t_medR, 'v2t_meanR': v2t_meanR
    }
    
    return metrics

def main(args):
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n-" * 80)
    print("\nLoading test configuration and data...")
    # Load dataset to get exact validation split matching training
    with open(args.matched_pairs, 'r') as f:
        matched_pairs = json.load(f)
        
    np.random.seed(args.seed)
    indices = np.random.permutation(len(matched_pairs))
    train_size = int(args.train_split * len(matched_pairs))
    val_indices = indices[train_size:]
    val_pairs = [matched_pairs[i] for i in val_indices]
    
    print(f"Total matched pairs: {len(matched_pairs)}")
    print(f"Validation pairs to test: {len(val_pairs)}")
    print("\n-" * 80)
    
    # Extract raw features 
    task_embs = np.array([p['task_embedding'] for p in val_pairs])
    visual_embs = np.array([p['visual_embedding'] for p in val_pairs])
    
    # 1. Evaluate cross-modal retrieval directly on the input features
    print("\n\n[Baseline] Raw Feature Retrieval Performance")
    print("\nHow well the initial Task Graphs & Visual clips match without fusion:")
    raw_metrics = evaluate_retrieval(task_embs, visual_embs)
    print(f" Task -> Video (T2V) | R@1: {raw_metrics['t2v_r1']:5.2f}% | R@5: {raw_metrics['t2v_r5']:5.2f}% | R@10: {raw_metrics['t2v_r10']:5.2f}% | MedR: {raw_metrics['t2v_medR']:.1f} | MeanR: {raw_metrics['t2v_meanR']:.1f}")
    print(f" Video -> Task (V2T) | R@1: {raw_metrics['v2t_r1']:5.2f}% | R@5: {raw_metrics['v2t_r5']:5.2f}% | R@10: {raw_metrics['v2t_r10']:5.2f}% | MedR: {raw_metrics['v2t_medR']:.1f} | MeanR: {raw_metrics['v2t_meanR']:.1f}")

    # 2. Evaluate using the learned Fusion Module
    model = FeatureFusionModule(
        embedding_dim=256,
        hidden_dim=512,
        output_dim=256,
        fusion_type=args.fusion_type
    ).to(device)
    
    checkpoint_path = os.path.join(args.model_dir, 'best_fusion_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"\n[!] Warning: Checkpoint not found at {checkpoint_path}. Train the model first.")
        return
        
    print(f"\n[Trained System] Validating Learned Fusion Network...")
    print(f"Loading weights from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict']) # if saved with dictionary
    else:
        model.load_state_dict(checkpoint) # if directly saved
        
    model.eval()

    val_dataset = MatchedPairsDataset(val_pairs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    all_fused = []
    all_fused_t = []
    all_fused_v = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            batch_task = batch['task_embedding'].to(device)
            batch_visual = batch['visual_embedding'].to(device)
            
            task_fused, visual_fused = model(batch_task, batch_visual)
            
            loss = contrastive_loss(task_fused, visual_fused, model.logit_scale)
            total_loss += loss.item()
            
            all_fused_t.append(task_fused.cpu().numpy())
            all_fused_v.append(visual_fused.cpu().numpy())
            
    val_loss = total_loss / len(val_loader)

    
    all_fused_t = np.concatenate(all_fused_t, axis=0)
    all_fused_v = np.concatenate(all_fused_v, axis=0)
    
    print("\n-" * 80)
    print("\n[Post-Fusion] Retrieval Performance")
    print("\nEvaluating Task->Video matching in the learned shared latent space:")
    post_metrics = evaluate_retrieval(all_fused_t, all_fused_v)
    print(f" Task -> Video (T2V) | R@1: {post_metrics['t2v_r1']:5.2f}% | R@5: {post_metrics['t2v_r5']:5.2f}% | R@10: {post_metrics['t2v_r10']:5.2f}% | MedR: {post_metrics['t2v_medR']:.1f} | MeanR: {post_metrics['t2v_meanR']:.1f}")
    print(f" Video -> Task (V2T) | R@1: {post_metrics['v2t_r1']:5.2f}% | R@5: {post_metrics['v2t_r5']:5.2f}% | R@10: {post_metrics['v2t_r10']:5.2f}% | MedR: {post_metrics['v2t_medR']:.1f} | MeanR: {post_metrics['v2t_meanR']:.1f}")
    
    improvement_r10 = post_metrics['t2v_r10'] - raw_metrics['t2v_r10']
    print(f"\n=> Absolute Improvement in T2V R@10 over Baseline: +{improvement_r10:.2f}%")
    print(f"=> Validation Loss: {val_loss:.4f}")
    
    print("\n-" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matched_pairs', type=str, default='outputs/matched_features_gt/matched_pairs.json', help='Matches file')
    parser.add_argument('--model_dir', type=str, default='outputs/fusion_model_gt', help='Path to check for fusion checkpoints')
    parser.add_argument('--train_split', type=float, default=0.8, help='Make sure this matches the training split')
    parser.add_argument('--seed', type=int, default=42, help='Make sure this matches the training random seed')
    parser.add_argument('--fusion_type', type=str, default='concat', help='Type of fusion (concat, gated, cross_attention)')
    parser.add_argument('--batch_size', type=int, default=128)
    
    args = parser.parse_args()
    main(args)
