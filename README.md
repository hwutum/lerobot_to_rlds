This is a dataset adaptor which converts lerobot dataset to rlds data format so we can use lerobot dataset for finetuning openVLA (https://github.com/moojink/openvla-oft)
The code gets inspiration from https://github.com/moojink/rlds_dataset_builder/tree/main
However the original code requires python<=3.9 which is imcompatible with lerobot (python >= 3.10)

Steps: 
(1) `python convert_lerobot_to_rlds.py` in a lerobot conda environment
(2) switch to rlds_dataset envoironment (see https://github.com/moojink/rlds_dataset_builder/tree/main) and run `tfds build --overwrite`
