#!/bin/bash

echo "🚀 Setting up Tencent InstantMesh..."

# 1. Clone the repository
if [ ! -d "InstantMesh" ]; then
    echo "📦 Cloning TencentARC/InstantMesh..."
    git clone https://github.com/TencentARC/InstantMesh.git
else
    echo "✅ InstantMesh folder already exists."
fi

# 2. Install Dependencies
# We explicitly force opencv-python-headless to a version compatible with NumPy 1.x
echo "⬇️  Fixing library versions..."
pip install "numpy<2.0.0" "opencv-python-headless<4.10.0"

echo "⬇️  Installing main dependencies..."
# We use --no-deps for requirements first to avoid it upgrading numpy back to 2.0
pip install -r InstantMesh/requirements.txt

# 3. Install helper tools
# Added 'onnxruntime-gpu' explicitly to fix the Import Error
pip install onnxruntime-gpu imageio[ffmpeg] trimesh rembg

# 4. Create a helper init file to make it importable
touch InstantMesh/__init__.py

echo "🎉 Setup complete! You can now run 'python generate_instantmesh.py'"