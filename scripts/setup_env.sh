#!/usr/bin/env bash
# ==============================================================================
# Setup script for MO-PPO Super Mario on Google Colab, Kaggle, and Linux
# Group 1 - Reinforcement Learning
# ==============================================================================

set -e

echo "=== [1/4] Installing system dependencies (ffmpeg, OpenGL, GLib) ==="
if command -v apt-get &> /dev/null; then
    apt-get update -qq
    apt-get install -y -qq ffmpeg libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
fi

echo "=== [2/4] Ensuring build tools and pip are up to date ==="
python -m pip install --upgrade pip setuptools wheel

echo "=== [3/4] Installing all dependencies from requirements.txt ==="
pip install -r requirements.txt

echo "=== [Verification] Testing core imports ==="
python -c "
from importlib.metadata import version
import numpy as np
import torch
import nes_py
import gym_super_mario_bros
import mo_gymnasium
import yaml
import cv2
print('=' * 50)
print(f'NumPy version        : {np.__version__}')
print(f'PyTorch version      : {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'NES-Py version       : {version(\"nes-py\")}')
print(f'Super Mario Bros     : {version(\"gym-super-mario-bros\")}')
print('ALL CORE LIBRARIES INSTALLED & IMPORTED SUCCESSFULLY!')
print('=' * 50)
"
