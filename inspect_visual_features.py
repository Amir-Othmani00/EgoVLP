import numpy as np

vf = np.load('visual_features/hiero_step_embeddings_256.npz', allow_pickle=True)

print("Keys in visual features file:", vf.files)
print("\nstep_embeddings shape:", vf['step_embeddings'].shape)
print("step_mask shape:", vf['step_mask'].shape)

try:
    print("\nvideo_ids type:", type(vf['video_ids']))
    print("video_ids sample:", vf['video_ids'][:5])
except Exception as e:
    print(f"Error reading video_ids: {e}")

try:
    print("\nlabels type:", type(vf['labels']))
    print("labels sample:", vf['labels'][:5])
except Exception as e:
    print(f"Error reading labels: {e}")
