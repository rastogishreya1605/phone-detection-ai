#!/usr/bin/env bash
# exit on error
set -o errexit

# Install System Dependencies (libGL install karne ke liye)
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install Python Dependencies
pip install -r requirements.txt