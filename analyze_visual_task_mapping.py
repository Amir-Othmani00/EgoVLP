import numpy as np
import json
from pathlib import Path

print("=" * 80)
print("Analyzing Visual Features and Boundaries")
print("=" * 80)

# Load boundaries
print("\nLoading hiero_step_boundaries.json...")
with open('visual_features/hiero_step_boundaries.json', 'r') as f:
    boundaries = json.load(f)

print(f"Type: {type(boundaries)}")
if isinstance(boundaries, dict):
    print(f"Keys: {list(boundaries.keys())[:10]}...")
    print(f"Total entries: {len(boundaries)}")
    
    # Show first entry
    first_key = list(boundaries.keys())[0]
    print(f"\nFirst entry (key: {first_key}):")
    print(json.dumps(boundaries[first_key], indent=2)[:500])
    
elif isinstance(boundaries, list):
    print(f"List length: {len(boundaries)}")
    print(f"First entry: {boundaries[0]}")

# Load visual features
print("\n" + "=" * 80)
print("Loading hiero_step_embeddings_256.npz...")
print("=" * 80)

vf = np.load('visual_features/hiero_step_embeddings_256.npz')

print(f"\nKeys: {vf.files}")

labels = vf['labels']
print(f"\nLabels shape: {labels.shape}")
print(f"Unique labels: {np.unique(labels)}")
print(f"Label distribution: {np.bincount(labels)}")

# Check if we can parse video_ids differently
print("\n" + "=" * 80)
print("Checking video_ids structure...")
print("=" * 80)

try:
    # Try different approaches to load video_ids
    vf_raw = np.load('visual_features/hiero_step_embeddings_256.npz', allow_pickle=True)
    video_ids = vf_raw['video_ids']
    print(f"Type of video_ids: {type(video_ids)}")
    print(f"Dtype: {video_ids.dtype if hasattr(video_ids, 'dtype') else 'N/A'}")
    
    if hasattr(video_ids, '__len__'):
        print(f"Length: {len(video_ids)}")
        print(f"First 5 items: {[str(x) for x in video_ids[:5]]}")
        
        # Check if they're strings (task names)
        if len(video_ids) > 0:
            first_item = video_ids[0]
            print(f"\nFirst item details:")
            print(f"  Type: {type(first_item)}")
            print(f"  Value: {first_item}")
            
            # Check if any item contains task names from annotations
            task_names = [
                'blenderbananapancakes', 'breakfastburritos', 'broccolistirfry', 
                'buttercorncup', 'capresebruschetta', 'cheesepimiento', 'coffee',
                'cucumberraita', 'dressedupmeatballs', 'herbomeletwithfriedtomatoes',
                'microwaveeggsandwich', 'microwavefrenchtoast', 'microwavemugpizza',
                'mugcake', 'panfriedtofu', 'pinwheels', 'ramen', 'sautedmushrooms',
                'scrambledeggs', 'spicedhotchocolate', 'spicytunaavocadowraps',
                'tomatochutney', 'tomatomozzarellasalad', 'zoodles'
            ]
            
            found_tasks = []
            for vid_id in video_ids[:20]:  # Check first 20
                vid_str = str(vid_id).lower()
                for task in task_names:
                    if task in vid_str:
                        found_tasks.append((vid_id, task))
                        break
            
            if found_tasks:
                print(f"\nFound task names in video_ids:")
                for vid, task in found_tasks[:5]:
                    print(f"  {vid} -> {task}")
            else:
                print(f"\nNo task names found in first 20 video_ids")
                print(f"Sample video_ids: {video_ids[:5]}")
    
except Exception as e:
    print(f"Error: {e}")

# Load task graph embeddings to compare
print("\n" + "=" * 80)
print("Comparing with task graph encodings...")
print("=" * 80)

tg = np.load('outputs/task_graph_encodings/task_graph_embeddings.npz')
print(f"Task graph tasks: {tg.files}")
print(f"Number of tasks: {len(tg.files)}")

# Load metadata
with open('outputs/task_graph_encodings/task_graph_metadata.json', 'r') as f:
    tg_metadata = json.load(f)

print(f"\nTask graph metadata keys: {list(tg_metadata.keys())[:5]}...")

# Summary
print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)
print(f"Visual features: 384 videos with labels 0,1")
print(f"Task graphs: {len(tg.files)} tasks")
print(f"\nHypothesis: Labels 0,1 are correctness indicators, not task labels")
print(f"Need to: Find task mapping in video_ids or boundaries")
