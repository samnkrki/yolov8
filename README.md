# Create a conda environment
conda create -n yolov8 python=3.8 -y

# Clone the ultralytics repository
git clone https://github.com/ultralytics/ultralytics

# Navigate to the cloned directory
cd ultralytics

# Install the package in editable mode for development
pip install -e .

# Install raytune for parameter tuning
pip install "ray[tune]"

# Install clearml
pip install clearml

# Initialize clearml
clearml-init

# Install pytorch and cudaconda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y