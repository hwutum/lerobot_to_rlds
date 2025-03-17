import h5py

episode_path = "/home/ke/Documents/lerobot_serl/lerobot/rlds_datasets/train/episode_0.h5"
with h5py.File(episode_path, "r") as F:
    breakpoint()
    F.keys()

print(actions.shape)
print(states.shape)
print(images.shape)
print(reward.shape)
print(language_instruction.shape)