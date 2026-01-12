import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class FeatureFusionModule(nn.Module):
    """
    Learnable projection to fuse task graph and visual features.
    """
    def __init__(self, embedding_dim=256, hidden_dim=512, output_dim=256, fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type
        self.embedding_dim = embedding_dim
        
        if fusion_type == 'concat':
            # Concatenate and project
            self.fusion = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        elif fusion_type == 'cross_attention':
            # Cross-attention mechanism
            self.query_proj = nn.Linear(embedding_dim, embedding_dim)
            self.key_proj = nn.Linear(embedding_dim, embedding_dim)
            self.value_proj = nn.Linear(embedding_dim, embedding_dim)
            self.out_proj = nn.Linear(embedding_dim, output_dim)
            self.norm = nn.LayerNorm(output_dim)
        elif fusion_type == 'gated':
            # Gated fusion
            self.gate = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Sigmoid()
            )
            self.proj = nn.Linear(embedding_dim, output_dim)
            self.norm = nn.LayerNorm(output_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, task_features, visual_features):
        """
        Args:
            task_features: (batch, embedding_dim)
            visual_features: (batch, embedding_dim)
        Returns:
            fused_features: (batch, output_dim)
        """
        if self.fusion_type == 'concat':
            combined = torch.cat([task_features, visual_features], dim=-1)
            fused = self.fusion(combined)
        elif self.fusion_type == 'cross_attention':
            # Task features attend to visual features
            Q = self.query_proj(task_features)
            K = self.key_proj(visual_features)
            V = self.value_proj(visual_features)
            
            # Scaled dot-product attention
            attn_weights = torch.softmax(Q @ K.T / np.sqrt(self.embedding_dim), dim=-1)
            attended = attn_weights @ V
            fused = self.norm(self.out_proj(attended))
        elif self.fusion_type == 'gated':
            # Gated combination
            gate_input = torch.cat([task_features, visual_features], dim=-1)
            gate = self.gate(gate_input)
            combined = gate * task_features + (1 - gate) * visual_features
            fused = self.norm(self.proj(combined))
        
        return fused


class MatchedPairsDataset(Dataset):
    """Dataset of matched task-visual pairs."""
    def __init__(self, matched_pairs):
        """
        Args:
            matched_pairs: list of dicts with keys:
                - 'task_embedding': list or numpy array
                - 'visual_embedding': list or numpy array
                - 'task_name': str
                - 'task_idx': int
                - 'visual_idx': int
        """
        self.pairs = matched_pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            'task_embedding': torch.FloatTensor(pair['task_embedding']),
            'visual_embedding': torch.FloatTensor(pair['visual_embedding']),
            'task_name': pair['task_name'],
            'task_idx': pair['task_idx'],
            'visual_idx': pair['visual_idx']
        }


def contrastive_loss(fused_features, task_features, visual_features, temperature=0.07):
    """
    Contrastive loss to ensure fused features are similar to both inputs.
    """
    # Normalize features
    fused_norm = nn.functional.normalize(fused_features, dim=-1)
    task_norm = nn.functional.normalize(task_features, dim=-1)
    visual_norm = nn.functional.normalize(visual_features, dim=-1)
    
    # Compute similarities
    sim_task = (fused_norm * task_norm).sum(dim=-1) / temperature
    sim_visual = (fused_norm * visual_norm).sum(dim=-1) / temperature
    
    # Loss: maximize similarity to both inputs
    loss = -torch.mean(sim_task + sim_visual)
    
    return loss


def train_epoch(model, train_loader, optimizer, device, temperature):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        task_emb = batch['task_embedding'].to(device)
        visual_emb = batch['visual_embedding'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        fused = model(task_emb, visual_emb)
        
        # Compute loss
        loss = contrastive_loss(fused, task_emb, visual_emb, temperature=temperature)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, device, temperature):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            task_emb = batch['task_embedding'].to(device)
            visual_emb = batch['visual_embedding'].to(device)
            
            fused = model(task_emb, visual_emb)
            loss = contrastive_loss(fused, task_emb, visual_emb, temperature=temperature)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main(args):
    print("=" * 80)
    print("Loading matched pairs...")
    print("=" * 80)
    
    # Load matched pairs
    with open(args.matched_pairs, 'r') as f:
        matched_pairs = json.load(f)
    
    print(f"Loaded {len(matched_pairs)} matched pairs")
    
    # Split into train/val
    np.random.seed(args.seed)
    indices = np.random.permutation(len(matched_pairs))
    train_size = int(args.train_split * len(matched_pairs))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_pairs = [matched_pairs[i] for i in train_indices]
    val_pairs = [matched_pairs[i] for i in val_indices]
    
    print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    
    # Create datasets and dataloaders
    train_dataset = MatchedPairsDataset(train_pairs)
    val_dataset = MatchedPairsDataset(val_pairs)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print("\n" + "=" * 80)
    print("Initializing fusion module...")
    print("=" * 80)
    
    # Get embedding dimension from first sample
    embedding_dim = len(matched_pairs[0]['task_embedding'])
    print(f"Embedding dimension: {embedding_dim}")
    
    # Initialize fusion module
    model = FeatureFusionModule(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        fusion_type=args.fusion_type
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Using device: {device}")
    print(f"\nFusion model architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    print("\n" + "=" * 80)
    print("Training...")
    print("=" * 80)
    
    best_loss = float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args.temperature)
        
        # Validate
        val_loss = validate(model, val_loader, device, args.temperature)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args),
                'embedding_dim': embedding_dim
            }
            torch.save(checkpoint, output_dir / 'best_fusion_model.pth')
            print(f'  ✓ Saved best model (val_loss: {val_loss:.4f})')
    
    # Save final model
    final_checkpoint = {
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'args': vars(args),
        'embedding_dim': embedding_dim
    }
    torch.save(final_checkpoint, output_dir / 'final_fusion_model.pth')
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Models saved to {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train fusion module on matched task-visual pairs')
    
    # Input
    parser.add_argument('--matched_pairs', type=str,
                        default='outputs/matched_features/matched_pairs.json',
                        help='Path to matched pairs JSON file')
    
    # Output
    parser.add_argument('--output_dir', type=str,
                        default='outputs/fusion_model',
                        help='Directory to save trained model')
    
    # Data split
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data to use for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Model architecture
    parser.add_argument('--fusion_type', type=str, default='concat',
                        choices=['concat', 'cross_attention', 'gated'],
                        help='Type of fusion module')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension for fusion module')
    parser.add_argument('--output_dim', type=int, default=256,
                        help='Output dimension for fused features')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    main(args)
