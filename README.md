# SO-101 机械臂操作手册

> SO-101 双臂系统：黑色臂 Leader（人手操控，7.4V），白色臂 Follower（执行动作，12V）

## 1. 数据转换（IsaacLab → LeRobot）

```bash
export LEROBOT_HOME=/home/quincylee/code/Robotics/data/arm/transformed
cd /home/quincylee/code/Robotics/leisaac

python scripts/convert/isaaclab2lerobotv3.py \
    --task_name=LeIsaac-SO101-PickOrange-v0 \
    --task_type=keyboard \
    --repo_id=quincylee/pick_orange_kb \
    --hdf5_root=/home/quincylee/code/Robotics/data/arm/recorded/pick_orange \
    --hdf5_files=kb_01.hdf5 \
    --device=cpu
```

## 2. 端口与权限

```bash
ls /dev/ttyACM*                          # 确定端口号
sudo chmod 666 /dev/ttyACM0              # Leader（黑色）
sudo chmod 666 /dev/ttyACM1              # Follower（白色）
```

## 3. 标定

```bash
# --- 自动标定 ---
# 黑色臂 Leader（人手操控，7.4V 电源）
python examples/calibrate/auto_calibrate_example.py \
    --port /dev/ttyACM0 --device-type tele

# 白色臂 Follower（执行动作，12V 电源）
python examples/calibrate/auto_calibrate_example.py \
    --port /dev/ttyACM1 --device-type robot

# --- 手动标定 ---
lerobot-calibrate \
    --teleop.type so101_leader --teleop.port /dev/ttyACM0 --teleop.id R07252802

lerobot-calibrate \
    --robot.type so101_follower --robot.port /dev/ttyACM1 --robot.id R12252802
```

## 4. 遥操作（协同测试）

```bash
# 单独测试 Leader
lerobot-teleoperate \
    --teleop.type so101_leader --teleop.port /dev/ttyACM0 --teleop.id R07252802

# 单独测试 Follower
lerobot-teleoperate \
    --robot.type so101_follower --robot.port /dev/ttyACM1 --robot.id R12252802

# Leader + Follower 协同
lerobot-teleoperate \
    --robot.type so101_follower --robot.port /dev/ttyACM1 --robot.id R12252802 \
    --teleop.type so101_leader --teleop.port /dev/ttyACM0 --teleop.id R07252802
```

## 5. 相机测试

```bash
python -c "
import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print('按 q 退出')
while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.imshow('camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()
"
```

## 6. 录制数据

```bash
conda activate lerobot
cd ~/code/Robotics

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=R12252802 \
    --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"fourcc":"MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=R07252802 \
    --dataset.repo_id=quincyyyy/so100_batch1 \
    --dataset.root=/home/quincylee/code/Robotics/data/arm/recorded/batch1 \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Pick and place" \
    --dataset.push_to_hub=false \
    --display_data=true
```

## 7. 复现录制动作（可选）

```bash
conda activate lerobot

lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=R12252802 \
    --dataset.repo_id=quincyyyy/so100_batch1 \
    --dataset.root=/home/quincylee/code/Robotics/data/arm/recorded/batch1 \
    --dataset.episode=0
```

## 8. 训练

### 本地单卡

```bash
conda activate lerobot
cd ~/code/Robotics/lerobot

lerobot-train \
    --dataset.repo_id=quincyyyy/so100_batch1 \
    --dataset.root=/home/quincylee/code/Robotics/data/arm/recorded/batch1 \
    --policy.type=smolvla \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/smolvla_batch1 \
    --job_name=smolvla_batch1 \
    --batch_size=64 \
    --steps=20000 \
    --wandb.enable=false
```

### V100 多卡

```bash
conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com

accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=batch1_enlarge \
  --dataset.root=/home/cxhlab/lqj_code/Arm_data/batch1_enlarge \
  --policy.path=/home/cxhlab/lqj_code/Robotics/smolvla \
  --policy.device=cuda \
  --policy.empty_cameras=2 \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --output_dir=/home/cxhlab/lqj_code/Robotics/smolvla/outputs/train/batch1_enlarge \
  --job_name=batch1_enlarge \
  --batch_size=8 \
  --steps=10000 \
  --save_freq=2500 \
  --log_freq=50
```

## 9. 推理部署

```bash
conda activate lerobot
cd ~/code/Robotics/lerobot

lerobot-rollout \
    --strategy.type=base \
    --policy.path=/home/quincylee/code/Robotics/data/models/pretrained_model_JIAXIAOQIU \
    --inference.type=rtc \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=R12252802 \
    --robot.cameras='{"camera1":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"fourcc":"MJPG"}}' \
    --task="Pick and place" \
    --duration=30
```

---








<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/latest_deps_tests.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/latest_deps_tests.yml?query=branch%3Amain)
[![Tests](https://github.com/huggingface/lerobot/actions/workflows/docker_publish.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/docker_publish.yml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/badge/Discord-Join_Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/q8Dzzpym3f)

</div>

**LeRobot** aims to provide models, datasets, and tools for real-world robotics in PyTorch. The goal is to lower the barrier to entry so that everyone can contribute to and benefit from shared datasets and pretrained models.

🤗 A hardware-agnostic, Python-native interface that standardizes control across diverse platforms, from low-cost arms (SO-100) to humanoids.

🤗 A standardized, scalable LeRobotDataset format (Parquet + MP4 or images) hosted on the Hugging Face Hub, enabling efficient storage, streaming and visualization of massive robotic datasets.

🤗 State-of-the-art policies that have been shown to transfer to the real-world ready for training and deployment.

🤗 Comprehensive support for the open-source ecosystem to democratize physical AI.

## Quick Start

LeRobot can be installed directly from PyPI.

```bash
pip install lerobot
lerobot-info
```

> [!IMPORTANT]
> For detailed installation guide, please see the [Installation Documentation](https://huggingface.co/docs/lerobot/installation).

## Robots & Control

<div align="center">
  <img src="./media/readme/robots_control_video.webp" width="640px" alt="Reachy 2 Demo">
</div>

LeRobot provides a unified `Robot` class interface that decouples control logic from hardware specifics. It supports a wide range of robots and teleoperation devices.

```python
from lerobot.robots.myrobot import MyRobot

# Connect to a robot
robot = MyRobot(config=...)
robot.connect()

# Read observation and send action
obs = robot.get_observation()
action = model.select_action(obs)
robot.send_action(action)
```

**Supported Hardware:** SO100, LeKiwi, Koch, HopeJR, OMX, EarthRover, Reachy2, Gamepads, Keyboards, Phones, OpenARM, Unitree G1.

While these devices are natively integrated into the LeRobot codebase, the library is designed to be extensible. You can easily implement the Robot interface to utilize LeRobot's data collection, training, and visualization tools for your own custom robot.

For detailed hardware setup guides, see the [Hardware Documentation](https://huggingface.co/docs/lerobot/integrate_hardware).

## LeRobot Dataset

To solve the data fragmentation problem in robotics, we utilize the **LeRobotDataset** format.

- **Structure:** Synchronized MP4 videos (or images) for vision and Parquet files for state/action data.
- **HF Hub Integration:** Explore thousands of robotics datasets on the [Hugging Face Hub](https://huggingface.co/lerobot).
- **Tools:** Seamlessly delete episodes, split by indices/fractions, add/remove features, and merge multiple datasets.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load a dataset from the Hub
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# Access data (automatically handles video decoding)
episode_index=0
print(f"{dataset[episode_index]['action'].shape=}\n")
```

Learn more about it in the [LeRobotDataset Documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

## SoTA Models

LeRobot implements state-of-the-art policies in pure PyTorch, covering Imitation Learning, Reinforcement Learning, and Vision-Language-Action (VLA) models, with more coming soon. It also provides you with the tools to instrument and inspect your training process.

<p align="center">
  <img alt="Gr00t Architecture" src="./media/readme/VLA_architecture.jpg" width="640px">
</p>

Training a policy is as simple as running a script configuration:

```bash
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

| Category                   | Models                                                                                                                                                                                                                  |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Imitation Learning**     | [ACT](./docs/source/policy_act_README.md), [Diffusion](./docs/source/policy_diffusion_README.md), [VQ-BeT](./docs/source/policy_vqbet_README.md), [Multitask DiT Policy](./docs/source/policy_multi_task_dit_README.md) |
| **Reinforcement Learning** | [HIL-SERL](./docs/source/hilserl.mdx), [TDMPC](./docs/source/policy_tdmpc_README.md) & QC-FQL (coming soon)                                                                                                             |
| **VLAs Models**            | [Pi0Fast](./docs/source/pi0fast.mdx), [Pi0.5](./docs/source/pi05.mdx), [GR00T N1.5](./docs/source/policy_groot_README.md), [SmolVLA](./docs/source/policy_smolvla_README.md), [XVLA](./docs/source/xvla.mdx)            |

Similarly to the hardware, you can easily implement your own policy & leverage LeRobot's data collection, training, and visualization tools, and share your model to the HF Hub

For detailed policy setup guides, see the [Policy Documentation](https://huggingface.co/docs/lerobot/bring_your_own_policies).

## Inference & Evaluation

Evaluate your policies in simulation or on real hardware using the unified evaluation script. LeRobot supports standard benchmarks like **LIBERO**, **MetaWorld** and more to come.

```bash
# Evaluate a policy on the LIBERO benchmark
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

Learn how to implement your own simulation environment or benchmark and distribute it from the HF Hub by following the [EnvHub Documentation](https://huggingface.co/docs/lerobot/envhub)

## Resources

- **[Documentation](https://huggingface.co/docs/lerobot/index):** The complete guide to tutorials & API.
- **[Chinese Tutorials: LeRobot+SO-ARM101中文教程-同济子豪兄](https://zihao-ai.feishu.cn/wiki/space/7589642043471924447)** Detailed doc for assembling, teleoperate, dataset, train, deploy. Verified by Seed Studio and 5 global hackathon players.
- **[Discord](https://discord.gg/q8Dzzpym3f):** Join the `LeRobot` server to discuss with the community.
- **[X](https://x.com/LeRobotHF):** Follow us on X to stay up-to-date with the latest developments.
- **[Robot Learning Tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial):** A free, hands-on course to learn robot learning using LeRobot.

## Citation

If you use LeRobot in your project, please cite the GitHub repository to acknowledge the ongoing development and contributors:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

If you are referencing our research or the academic paper, please also cite our ICLR publication:

<details>
<summary><b>ICLR 2026 Paper</b></summary>

```bibtex
@inproceedings{cadenelerobot,
  title={LeRobot: An Open-Source Library for End-to-End Robot Learning},
  author={Cadene, Remi and Alibert, Simon and Capuano, Francesco and Aractingi, Michel and Zouitine, Adil and Kooijmans, Pepijn and Choghari, Jade and Russi, Martino and Pascal, Caroline and Palma, Steven and Shukor, Mustafa and Moss, Jess and Soare, Alexander and Aubakirova, Dana and Lhoest, Quentin and Gallou\'edec, Quentin and Wolf, Thomas},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2602.22818}
}
```

</details>

## Contribute

We welcome contributions from everyone in the community! To get started, please read our [CONTRIBUTING.md](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md) guide. Whether you're adding a new feature, improving documentation, or fixing a bug, your help and feedback are invaluable. We're incredibly excited about the future of open-source robotics and can't wait to work with you on what's next—thank you for your support!

<p align="center">
  <img alt="SO101 Video" src="./media/readme/so100_video.webp" width="640px">
</p>

<div align="center">
<sub>Built by the <a href="https://huggingface.co/lerobot">LeRobot</a> team at <a href="https://huggingface.co">Hugging Face</a> with ❤️</sub>
</div>
