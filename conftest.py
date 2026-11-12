"""Pytest configuration for MO-PPO Mario tests."""

import os
import sys

# Ensure root workspace directory is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
