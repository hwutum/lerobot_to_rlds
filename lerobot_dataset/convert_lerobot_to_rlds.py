#!/usr/bin/env python

"""
Script to convert a LeRobot dataset to RLDS format and save it as a TensorFlow dataset.

Example usage:
python convert_lerobot_to_rlds.py \
    --repo-id hangwu/piper_joint_ep_20250422_realsense \
    --output-dir /home/tars/Datasets/piper_joint_rlds/
"""

import argparse
import os
import sys
import tensorflow_datasets as tfds
import numpy as np
import torch
import traceback  # Add this to get detailed error information
from tqdm import tqdm  # Add this import for the progress bar
import h5py
# Add the path to the LeRobot library
# Adjust this path to point to the directory containing the lerobot package
sys.path.append('/home/tars/Code/lerobot')  # Update this path to where your lerobot code is located

# Function to load your dataset
def load_dataset_and_save_to_disk(repo_id, root=None, output_dir=None):
    """
    Load a dataset by repo_id using LeRobotDataset.
    
    Args:
        repo_id: Repository ID of the dataset
        root: Root directory for the dataset stored locally
        
    Returns:
        A dataset object in the format expected by DatasetToRLDSConverter
    """
    try:
        # Try to import LeRobotDataset
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            
            # Load the LeRobot dataset
            lerobot_dataset = LeRobotDataset(repo_id, root=root)
            
            # Create a dataset structure compatible with our converter
            dataset = {
                "episodes": [],
                "metadata": {
                    "repo_id": repo_id
                }
            }
            
            # Define EpisodeSampler class (similar to the one in visualize_dataset.py)
            class EpisodeSampler(torch.utils.data.Sampler):
                def __init__(self, dataset, episode_index):
                    from_idx = dataset.episode_data_index["from"][episode_index].item()
                    to_idx = dataset.episode_data_index["to"][episode_index].item()
                    self.frame_ids = range(from_idx, to_idx)
                    
                def __iter__(self):
                    return iter(self.frame_ids)
                    
                def __len__(self):
                    return len(self.frame_ids)
            
            # Function to convert CHW float32 tensor to HWC uint8 numpy array
            def to_hwc_uint8_numpy(chw_float32_torch):
                if chw_float32_torch.dtype == torch.float32 and chw_float32_torch.ndim == 3:
                    c, h, w = chw_float32_torch.shape
                    if c == 3:  # RGB image
                        hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
                        return hwc_uint8_numpy
                # Return as is if not a float32 CHW image
                return chw_float32_torch.numpy()
            
            # Get number of episodes
            num_episodes = len(lerobot_dataset.episode_data_index["from"])
            
            # Process each episode
            for episode_idx in tqdm(range(num_episodes)):
                # Create a sampler for this episode
                episode_sampler = EpisodeSampler(lerobot_dataset, episode_idx)
                
                # Create a dataloader with batch size 1 to load frames one by one
                dataloader = torch.utils.data.DataLoader(
                    lerobot_dataset,
                    batch_size=1,
                    sampler=episode_sampler,
                    num_workers=0
                )
                
                # Create episode structure
                episode = {
                    "frames": [],
                    "file_path": f"episode_{episode_idx}"
                }
                
                # Extract language instruction if available
                language_instruction = ""
                if hasattr(lerobot_dataset, 'get_language_instruction'):
                    language_instruction = lerobot_dataset.get_language_instruction(episode_idx)
                
                # Process each frame in the episode
                for batch in dataloader:
                    # Convert batch tensors to numpy arrays and remove batch dimension
                    frame = {}
                    
                    # Add state if available
                    if 'observation.state' in batch:
                        frame['state'] = batch['observation.state'][0].numpy()
                    
                    # Add action if available
                    if 'action' in batch:
                        frame['action'] = batch['action'][0].numpy()
                    
                    # Add reward if available
                    if 'next.reward' in batch:
                        frame['reward'] = batch['next.reward'][0].item()
                    
                    # Add camera images
                    for key in lerobot_dataset.meta.camera_keys:
                        if key in batch:
                            frame[key.replace('.', '_')] = to_hwc_uint8_numpy(batch[key][0])
                    
                    # Add language instruction
                    if language_instruction:
                        frame['language_instruction'] = language_instruction
                    else:
                        frame['language_instruction'] = ""
                    
                    episode["frames"].append(frame)
                
                # save episode to disk
                # Save episode to disk using a more memory-efficient format
                os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
                
                # Use HDF5 format
                with h5py.File(os.path.join(output_dir, f"train/episode_{episode_idx}.h5"), 'w') as f:
                    # Store metadata
                    f.attrs['file_path'] = episode['file_path']
                    
                    # Create a group for frames data
                    frames_group = f.create_group('frames')
                    
                    # Get number of frames
                    num_frames = len(episode['frames'])
                    
                    # Initialize datasets for the first frame to get shapes
                    first_frame = episode['frames'][0]
                    
                    # Create datasets for each data type in the first frame
                    for key, value in first_frame.items():
                        if isinstance(value, np.ndarray):
                            # Create dataset with appropriate shape and type
                            frames_group.create_dataset(
                                key, 
                                shape=(num_frames, *value.shape),
                                dtype=value.dtype
                            )
                        elif isinstance(value, (int, float)):
                            # Handle scalar values
                            frames_group.create_dataset(
                                key,
                                shape=(num_frames,),
                                dtype=np.float32 if isinstance(value, float) else np.int32
                            )
                        elif isinstance(value, str):
                            # For string values like language_instruction, create variable-length string dataset
                            frames_group.create_dataset(
                                key,
                                shape=(num_frames,),
                                dtype=h5py.special_dtype(vlen=str)
                            )
                    
                    # Fill in the datasets
                    for frame_idx, frame in enumerate(episode['frames']):
                        for key, value in frame.items():
                            if key in frames_group:
                                if isinstance(value, (np.ndarray, int, float, str)):
                                    frames_group[key][frame_idx] = value
                                else:
                                    # Convert other types to string if needed
                                    frames_group[key][frame_idx] = str(value)
                    
                    # Add camera keys as a list attribute for reference
                    if hasattr(lerobot_dataset, 'meta') and hasattr(lerobot_dataset.meta, 'camera_keys'):
                        camera_keys = [key.replace('.', '_') for key in lerobot_dataset.meta.camera_keys]
                        f.attrs['camera_keys'] = np.array(camera_keys, dtype=h5py.special_dtype(vlen=str))
            
            #     # Add episode to dataset
            #     dataset["episodes"].append(episode)
            
            # return dataset
            
        except ImportError:
            print(f"Warning: Could not import LeRobotDataset")
            print("Attempting to load dataset directly from HuggingFace...")
            
            # Alternative implementation using huggingface datasets
            import datasets
            hf_dataset = datasets.load_dataset(repo_id)
            
            # Convert HuggingFace dataset to the expected format
            dataset = {
                "episodes": [],
                "metadata": {
                    "repo_id": repo_id
                }
            }
            
            # Process the HuggingFace dataset
            # This will depend on the structure of your dataset
            # ...
            
            return dataset
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(traceback.format_exc())
        print("Falling back to dummy dataset")
        
        # Create a minimal dataset with one empty episode for testing
        return {
            "episodes": [{
                "frames": [{
                    "action": np.zeros(1, dtype=np.float32),
                    "state": np.zeros(1, dtype=np.float32),
                    "image": np.zeros((64, 64, 3), dtype=np.uint8),
                    "language_instruction": "test instruction"
                }],
                "file_path": "test_path"
            }],
            "metadata": {
                "repo_id": repo_id
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to RLDS format")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of HuggingFace repository containing a LeRobotDataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the RLDS dataset. If not provided, uses the default TFDS data directory.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory for the dataset stored locally.",
    )
    
    args = parser.parse_args()
    # Load the dataset
    dataset = load_dataset_and_save_to_disk(args.repo_id, args.root, args.output_dir)
    

if __name__ == "__main__":
    main() 