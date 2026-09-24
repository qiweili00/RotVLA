# Data Preparation

All datasets must be converted to **LeRobot v3.0 format** before use. After conversion, update the `data_info` paths in the YAML config files under `core/config/`.

---

## Finetuning Datasets

### [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

Download the four LIBERO suites and convert them to LeRobot v3.0 format using [any4lerobot](https://github.com/Tavish9/any4lerobot) 

Or download the converted [datasets](https://huggingface.co/datasets/QiweiLi/LIBERO-v3).


### [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)

Download the RoboTwin 2.0 dataset and convert to LeRobot v3.0 format. Conversion scripts are provided by [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/policy/GO1/README.md)

Or download the converted [datasets](https://huggingface.co/datasets/QiweiLi/RoboTwin2-v3).

---

## Pretraining Datasets

### [Open X-Embodiment](https://robotics-transformer-x.github.io/)

We use 22 datasets from Open X-Embodiment. Data is originally in RLDS format.

**Conversion pipeline:**
1. Download raw RLDS data.
2. Convert RLDS → LeRobot v3.0 using [any4lerobot](https://github.com/Tavish9/any4lerobot)

### [AgiBot World Alpha](https://huggingface.co/datasets/Traly/AgiBotWorld-Alpha-lerobot)

Available directly in LeRobot format on HuggingFace:
- [Traly/AgiBotWorld-Alpha-lerobot](https://huggingface.co/datasets/Traly/AgiBotWorld-Alpha-lerobot)
- [weijian-sun/agibotworld-lerobot](https://huggingface.co/datasets/weijian-sun/agibotworld-lerobot)

> **Note:** We recommend creating a new `info.json` for each task that removes fisheye camera entries, as they significantly slow down data loading without contributing to training. Detailed in [`core/dataset/lerobot_dataset_pretrain_mp.py`](../core/dataset/lerobot_dataset_pretrain_mp.py) Line968.

### [RoboMIND](https://huggingface.co/collections/BAAI-DataCube/robomind-lerobot-v30-431-task-collection)

Available in LeRobot v3.0 format on HuggingFace. After downloading, merge datasets that share the same robot platform using the merge script in `data/merge_by_metadata.txt`.

### [RoboCOIN](https://flagopen.github.io/RoboCOIN-DataManager/)

Originally in LeRobot v2.0 format. Convert to v3.0 using [any4lerobot](https://github.com/Tavish9/any4lerobot), then merge datasets that share the same robot platform using the merge script in `data/merge_by_metadata_robocoin.txt`.

### [Ego4D](https://ego4d-data.org/)

Egocentric human video data. We follow the Ego4D processing in [UniVLA](https://github.com/OpenDriveLab/UniVLA) — refer to their instructions, then convert the result to LeRobot v3.0 format.


