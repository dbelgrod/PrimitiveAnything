# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

PrimitiveAnything is a research project that converts 3D meshes/point clouds into assemblies of primitive shapes (cubes, spheres, cylinders) using an auto-regressive transformer. The system takes a 3D model as input and outputs a set of transformed primitives that approximate the original shape.

**Paper**: [arXiv:2505.04622](https://arxiv.org/abs/2505.04622)

## Architecture

- **Model**: `PrimitiveTransformerDiscrete` - Auto-regressive transformer that predicts primitive parameters sequentially
- **Shape Conditioning**: Uses Michelangelo's point cloud encoder for shape understanding
- **Primitives**: Three types - CubeBevel (0), SphereSharp (1), CylinderSharp (2)
- **Parameters per primitive**: Scale (3D), Rotation (Euler XYZ), Translation (3D), Type code

## Key Commands

```bash
# Setup environment
conda create -n primitiveanything python=3.9 -y
conda activate primitiveanything
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Run demo on 3D files (GLB, OBJ, etc.)
python demo.py --input ./data/demo_glb --log_path ./results/demo

# Run inference on test set
python infer.py

# Sample point clouds from predictions
python sample.py

# Evaluate metrics (Chamfer, EMD, Hausdorff, IoU)
python eval.py
```

## Project Structure

```
├── demo.py                     # Demo script for 3D file inference
├── infer.py                    # Test set inference
├── sample.py                   # Point cloud sampling from predictions
├── eval.py                     # Evaluation metrics calculation
├── configs/
│   └── infer.yml               # Model and dataset configuration
├── primitive_anything/
│   ├── primitive_transformer.py  # Main transformer model
│   ├── primitive_dataset.py      # Dataset handling
│   ├── michelangelo/             # Shape conditioning encoder
│   └── utils/                    # Utility functions
├── data/                       # (gitignored) Input data
│   ├── basic_shapes_norm/      # Normalized primitive meshes
│   ├── basic_shapes_norm_pc10000/  # Primitive point clouds
│   ├── demo_glb/               # Demo files
│   └── test_pc/                # Test point clouds
├── ckpt/                       # (gitignored) Model checkpoints
│   ├── mesh-transformer.ckpt.60.pt  # Main model
│   └── shapevae-256.ckpt       # Michelangelo encoder
└── results/                    # (gitignored) Output directory
```

## Model Configuration

Key parameters from `configs/infer.yml`:
- **Discretization bins**: Scale=128, Rotation=181, Translation=128
- **Value ranges**: Scale=[0,1], Rotation=[-180,180], Translation=[-1,1]
- **Model dimensions**: dim=768, attn_depth=6, attn_heads=6
- **Max primitives per model**: 144

## Dependencies

Core dependencies: PyTorch 2.1.0, PyTorch3D, trimesh, open3d, transformers, accelerate, x-transformers

Note: PyTorch3D requires installation from GitHub: `pip install git+https://github.com/facebookresearch/pytorch3d.git`

## Common Tasks

### Processing custom 3D models
```python
python demo.py --input /path/to/model.glb --log_path ./results/custom
```

### Modifying generation behavior
- Edit `temperature` parameter in demo.py/infer.py (0.0 = greedy, higher = more random)
- `dilated_offset` and `do_marching_cubes` control preprocessing for thin structures
