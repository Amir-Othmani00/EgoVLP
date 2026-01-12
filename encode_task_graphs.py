import os
import json
import argparse
import torch
import transformers
from pathlib import Path
import numpy as np

from model.model import FrozenInTime
from parse_config import ConfigParser


def load_task_graphs(task_graphs_dir):
    """Load all task graph JSON files and extract text descriptions."""
    task_graphs = {}
    task_graphs_path = Path(task_graphs_dir)
    
    for json_file in sorted(task_graphs_path.glob('*.json')):
        with open(json_file, 'r') as f:
            data = json.load(f)
            task_name = json_file.stem
            
            # Extract step descriptions (excluding START and END)
            steps = data.get('steps', {})
            descriptions = []
            for step_id, step_text in steps.items():
                if step_text not in ['START', 'END']:
                    descriptions.append(step_text)
            
            task_graphs[task_name] = {
                'descriptions': descriptions,
                'steps': steps,
                'edges': data.get('edges', [])
            }
    
    return task_graphs


def encode_texts(model, tokenizer, texts, device, batch_size=32):
    """Encode a list of texts using the EgoVLP text encoder."""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize the texts
            text_data = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=77  # typical CLIP-style max length
            )
            
            # Move to device
            text_data = {key: val.to(device) for key, val in text_data.items()}
            
            # Encode using the model's text encoder
            text_embeddings = model.compute_text(text_data)
            
            all_embeddings.append(text_embeddings.cpu().numpy())
    
    # Concatenate all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings


def main(args):
    # Load config (you can use a pre-trained config or create a minimal one)
    if args.config:
        config = ConfigParser.from_args(args, options=['config'])
    else:
        # Create minimal config for loading model
        config_dict = {
            'arch': {
                'type': 'FrozenInTime',
                'args': {
                    'video_params': {
                        'model': 'SpaceTimeTransformer',
                        'arch_config': 'base_patch16_224',
                        'num_frames': 16,
                        'pretrained': True,
                        'time_init': 'zeros'
                    },
                    'text_params': {
                        'model': 'distilbert-base-uncased',
                        'pretrained': True,
                        'input': 'text'
                    },
                    'projection_dim': 256,
                    'load_checkpoint': args.checkpoint,
                    'projection': 'minimal',
                    'load_temporal_fix': 'zeros'
                }
            }
        }
        
        # Create a simple ConfigParser-like object
        class SimpleConfig:
            def __init__(self, config):
                self._config = config
                
            def initialize(self, name, module):
                module_args = self._config[name]['args']
                return getattr(module, self._config[name]['type'])(**module_args)
            
            def __getitem__(self, key):
                return self._config[key]
        
        config = SimpleConfig(config_dict)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize tokenizer
    text_model_name = config['arch']['args']['text_params']['model']
    tokenizer = transformers.AutoTokenizer.from_pretrained(text_model_name)
    print(f'Loaded tokenizer: {text_model_name}')
    
    # Build model
    import model.model as module_arch
    model = config.initialize('arch', module_arch)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f'Loading checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Fix state dict if needed (for DataParallel models)
        from utils.util import state_dict_data_parallel_fix
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        model.load_state_dict(new_state_dict, strict=False)
        print('Checkpoint loaded successfully')
    
    model = model.to(device)
    model.eval()
    
    # Load task graphs
    print(f'Loading task graphs from: {args.task_graphs_dir}')
    task_graphs = load_task_graphs(args.task_graphs_dir)
    print(f'Loaded {len(task_graphs)} task graphs')
    
    # Encode all descriptions
    all_encodings = {}
    
    for task_name, task_data in task_graphs.items():
        print(f'\nEncoding task: {task_name}')
        descriptions = task_data['descriptions']
        print(f'  Number of steps: {len(descriptions)}')
        
        if descriptions:
            # Encode the descriptions
            embeddings = encode_texts(model, tokenizer, descriptions, device, batch_size=args.batch_size)
            
            all_encodings[task_name] = {
                'embeddings': embeddings,
                'descriptions': descriptions,
                'steps': task_data['steps'],
                'edges': task_data['edges']
            }
            
            print(f'  Embeddings shape: {embeddings.shape}')
    
    # Save the encodings
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy archive
    output_file = output_path / 'task_graph_embeddings.npz'
    np.savez_compressed(
        output_file,
        **{task_name: data['embeddings'] for task_name, data in all_encodings.items()}
    )
    print(f'\nSaved embeddings to: {output_file}')
    
    # Also save metadata
    metadata = {
        task_name: {
            'descriptions': data['descriptions'],
            'steps': data['steps'],
            'edges': data['edges'],
            'embedding_shape': data['embeddings'].shape
        }
        for task_name, data in all_encodings.items()
    }
    
    metadata_file = output_path / 'task_graph_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'Saved metadata to: {metadata_file}')
    
    return all_encodings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode task graph descriptions using EgoVLP text encoder')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional)')
    parser.add_argument('--checkpoint', type=str, default='./pretrained/egovlp.pth',
                        help='Path to pre-trained EgoVLP checkpoint')
    parser.add_argument('--task_graphs_dir', type=str, 
                        default='annotations/task_graphs',
                        help='Directory containing task graph JSON files')
    parser.add_argument('--output_dir', type=str, 
                        default='outputs/task_graph_encodings',
                        help='Directory to save encoded embeddings')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding')
    
    args = parser.parse_args()
    
    main(args)
