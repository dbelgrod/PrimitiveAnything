#!/usr/bin/env python3
"""
PrimitiveAnything Web App

Upload a GLB/OBJ file and convert it to primitives.
Shows original and primitive models side by side with download options.

Usage:
    python primitive_web.py
    Open http://localhost:5002 in your browser
"""

import os
import sys
import time
import glob
import json
import uuid
from pathlib import Path
from datetime import datetime

try:
    from flask import Flask, render_template_string, request, jsonify, send_from_directory
except ImportError:
    print("Error: flask package not installed.")
    print("Install with: pip3 install flask")
    sys.exit(1)

# Check if running with GPU support
GPU_AVAILABLE = False
MODEL_LOADED = False
DEVICE = None
transformer = None
accelerator = None
mesh_bs = None

def try_load_model():
    """Attempt to load the PrimitiveAnything model."""
    global GPU_AVAILABLE, MODEL_LOADED, DEVICE, transformer, accelerator, mesh_bs

    try:
        import torch

        # Check for available GPU backends
        if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
            GPU_AVAILABLE = True
            print("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            DEVICE = torch.device('mps')
            GPU_AVAILABLE = True
            print("Using Apple MPS (Metal)")
        else:
            DEVICE = torch.device('cpu')
            GPU_AVAILABLE = False
            print("Warning: No GPU available. Using CPU (will be slow).")

        if not GPU_AVAILABLE:
            print("The app will run in demo mode (upload only, no conversion).")
            return False

        import yaml
        import trimesh
        import numpy as np
        from accelerate import Accelerator
        from primitive_anything.primitive_transformer import PrimitiveTransformerDiscrete
        from primitive_anything.utils import count_parameters
        from primitive_anything.utils.logger import print_log

        # Note: Don't set PYOPENGL_PLATFORM on Mac - it doesn't have EGL
        # On Linux headless servers, you would use: os.environ['PYOPENGL_PLATFORM'] = 'egl'
        import platform
        if platform.system() != 'Darwin':
            os.environ['PYOPENGL_PLATFORM'] = 'egl'

        # Config paths
        bs_dir = 'data/basic_shapes_norm'
        config_path = './configs/infer.yml'
        checkpoint_path = './ckpt/mesh-transformer.ckpt.60.pt'

        # Check if files exist
        if not os.path.exists(config_path):
            print(f"Warning: Config not found at {config_path}")
            return False
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            return False

        # Load basic shapes
        mesh_bs = {}
        for bs_path in glob.glob(os.path.join(bs_dir, '*.ply')):
            bs_name = os.path.basename(bs_path)
            bs = trimesh.load(bs_path)
            bs.visual.uv = np.clip(bs.visual.uv, 0, 1)
            bs.visual = bs.visual.to_color()
            mesh_bs[bs_name] = bs

        if not mesh_bs:
            print(f"Warning: No basic shapes found in {bs_dir}")
            return False

        # Load config
        with open(config_path, 'r') as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)

        # Create model
        model_cfg = config['model'].copy()
        model_cfg.pop('name')
        transformer = PrimitiveTransformerDiscrete(**model_cfg)

        # Load checkpoint (weights_only=False needed for older checkpoints with numpy arrays)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        transformer.load_state_dict(checkpoint)

        # Prepare with accelerator (skip for MPS as accelerate has issues with it)
        if DEVICE.type == 'cuda':
            accelerator = Accelerator(mixed_precision='fp16')
            transformer = accelerator.prepare(transformer)
        else:
            # For MPS/CPU, just move to device
            transformer = transformer.to(DEVICE)
            accelerator = None

        transformer.eval()
        transformer.bs_pc = transformer.bs_pc.to(DEVICE)
        transformer.rotation_matrix_align_coord = transformer.rotation_matrix_align_coord.to(DEVICE)

        MODEL_LOADED = True
        print("✓ PrimitiveAnything model loaded successfully!")
        return True

    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("The app will run in demo mode.")
        return False


app = Flask(__name__)
OUTPUT_DIR = Path("./primitive_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = Path("./primitive_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Shape mappings
CODE_SHAPE = {
    0: 'SM_GR_BS_CubeBevel_001.ply',
    1: 'SM_GR_BS_SphereSharp_001.ply',
    2: 'SM_GR_BS_CylinderSharp_001.ply',
}

SHAPENAME_MAP = {
    'SM_GR_BS_CubeBevel_001.ply': 1101002001034001,
    'SM_GR_BS_SphereSharp_001.ply': 1101002001034010,
    'SM_GR_BS_CylinderSharp_001.ply': 1101002001034002,
}


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PrimitiveAnything - GLB to Primitives</title>
    <script src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js" type="module"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #eee;
        }
        .header {
            text-align: center;
            padding: 30px 20px;
            background: rgba(0,0,0,0.3);
        }
        h1 {
            color: #00d4ff;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #888;
            font-size: 1.1em;
        }
        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            margin-top: 15px;
        }
        .status-ready { background: #00c853; color: #000; }
        .status-demo { background: #ff9800; color: #000; }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px;
        }

        .upload-section {
            background: #16213e;
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            border: 2px dashed #0f3460;
            transition: border-color 0.3s, background 0.3s;
        }
        .upload-section:hover {
            border-color: #00d4ff;
            background: #1a2744;
        }
        .upload-section.dragover {
            border-color: #00d4ff;
            background: #1a2744;
        }

        .upload-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }

        input[type="file"] { display: none; }

        .upload-btn {
            display: inline-block;
            padding: 15px 40px;
            background: linear-gradient(135deg, #00d4ff, #0077b6);
            color: white;
            border: none;
            border-radius: 30px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 25px rgba(0, 212, 255, 0.4);
        }

        .supported-formats {
            margin-top: 20px;
            color: #666;
            font-size: 0.9em;
        }

        .viewers-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 30px;
        }
        @media (max-width: 900px) {
            .viewers-container { grid-template-columns: 1fr; }
        }

        .viewer-panel {
            background: #16213e;
            border-radius: 16px;
            overflow: hidden;
        }
        .viewer-header {
            padding: 15px 20px;
            background: #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .viewer-title {
            font-weight: 600;
            color: #00d4ff;
        }
        .download-btn {
            padding: 8px 20px;
            background: #00d4ff;
            color: #000;
            border: none;
            border-radius: 20px;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            font-size: 0.9em;
        }
        .download-btn:hover {
            background: #00b8d9;
        }
        .download-btn:disabled {
            background: #444;
            color: #888;
            cursor: not-allowed;
        }

        model-viewer {
            width: 100%;
            height: 400px;
            background: #0a0a1a;
        }
        .viewer-placeholder {
            width: 100%;
            height: 400px;
            background: #0a0a1a;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #444;
            font-size: 1.2em;
        }

        .stats-panel {
            background: #16213e;
            border-radius: 16px;
            padding: 25px;
            margin-top: 30px;
        }
        .stats-title {
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        .stat-item {
            background: #0f3460;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value {
            font-size: 1.8em;
            font-weight: 700;
            color: #00d4ff;
        }
        .stat-label {
            color: #888;
            font-size: 0.9em;
            margin-top: 5px;
        }

        .progress-container {
            display: none;
            margin-top: 30px;
        }
        .progress-container.active { display: block; }
        .progress-bar {
            height: 8px;
            background: #0f3460;
            border-radius: 4px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #0077b6);
            width: 0%;
            transition: width 0.3s;
        }
        .progress-text {
            text-align: center;
            margin-top: 10px;
            color: #888;
        }

        .genesis-code {
            background: #16213e;
            border-radius: 16px;
            padding: 25px;
            margin-top: 30px;
            display: none;
        }
        .genesis-code.active { display: block; }
        .genesis-code h3 {
            color: #00d4ff;
            margin-bottom: 15px;
        }
        .code-block {
            background: #0a0a1a;
            padding: 20px;
            border-radius: 10px;
            overflow-x: auto;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9em;
            line-height: 1.6;
        }
        .code-block .keyword { color: #ff79c6; }
        .code-block .string { color: #f1fa8c; }
        .code-block .number { color: #bd93f9; }
        .code-block .comment { color: #6272a4; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔷 PrimitiveAnything</h1>
        <p class="subtitle">Convert 3D models to physics-ready primitives</p>
        <span class="status-badge {{ 'status-ready' if model_loaded else 'status-demo' }}">
            {{ '✓ Model Ready' if model_loaded else '⚠ Demo Mode (No GPU)' }}
        </span>
    </div>

    <div class="container">
        <div class="upload-section" id="dropZone">
            <div class="upload-icon">📦</div>
            <h2>Drop your 3D model here</h2>
            <p style="margin: 15px 0; color: #888;">or</p>
            <label class="upload-btn">
                Choose File
                <input type="file" id="fileInput" accept=".glb,.gltf,.obj,.stl,.ply">
            </label>
            <p class="supported-formats">Supported: GLB, GLTF, OBJ, STL, PLY</p>
        </div>

        <div class="progress-container" id="progressContainer">
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <p class="progress-text" id="progressText">Processing...</p>
        </div>

        <div class="viewers-container" id="viewersContainer" style="display: none;">
            <div class="viewer-panel">
                <div class="viewer-header">
                    <span class="viewer-title">Original Model</span>
                    <a class="download-btn" id="downloadOriginal" href="#" download>Download GLB</a>
                </div>
                <div id="originalViewer">
                    <div class="viewer-placeholder">Original model will appear here</div>
                </div>
            </div>

            <div class="viewer-panel">
                <div class="viewer-header">
                    <span class="viewer-title">Primitive Model</span>
                    <a class="download-btn" id="downloadPrimitive" href="#" download>Download GLB</a>
                </div>
                <div id="primitiveViewer">
                    <div class="viewer-placeholder">Primitive model will appear here</div>
                </div>
            </div>
        </div>

        <div class="stats-panel" id="statsPanel" style="display: none;">
            <h3 class="stats-title">Conversion Statistics</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value" id="statPrimitives">-</div>
                    <div class="stat-label">Primitives</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="statCubes">-</div>
                    <div class="stat-label">Cubes</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="statSpheres">-</div>
                    <div class="stat-label">Spheres</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="statCylinders">-</div>
                    <div class="stat-label">Cylinders</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="statTime">-</div>
                    <div class="stat-label">Time (sec)</div>
                </div>
            </div>
        </div>

        <div class="genesis-code" id="genesisCode">
            <h3>🎮 Genesis Physics Code</h3>
            <p style="color: #888; margin-bottom: 15px;">Copy this code to use the primitives in Genesis simulation:</p>
            <div class="code-block" id="codeBlock">
                <span class="comment"># Genesis code will appear here after conversion</span>
            </div>
        </div>
    </div>

    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const progressContainer = document.getElementById('progressContainer');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const viewersContainer = document.getElementById('viewersContainer');
        const statsPanel = document.getElementById('statsPanel');
        const genesisCode = document.getElementById('genesisCode');

        // Drag and drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
            dropZone.addEventListener(event, e => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        ['dragenter', 'dragover'].forEach(event => {
            dropZone.addEventListener(event, () => dropZone.classList.add('dragover'));
        });

        ['dragleave', 'drop'].forEach(event => {
            dropZone.addEventListener(event, () => dropZone.classList.remove('dragover'));
        });

        dropZone.addEventListener('drop', e => {
            const file = e.dataTransfer.files[0];
            if (file) processFile(file);
        });

        fileInput.addEventListener('change', e => {
            const file = e.target.files[0];
            if (file) processFile(file);
        });

        async function processFile(file) {
            // Show progress
            progressContainer.classList.add('active');
            progressFill.style.width = '10%';
            progressText.textContent = 'Uploading file...';

            const formData = new FormData();
            formData.append('file', file);

            try {
                // Upload and process
                progressFill.style.width = '30%';
                progressText.textContent = 'Processing model...';

                const response = await fetch('/api/convert', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (!result.success) {
                    throw new Error(result.error);
                }

                progressFill.style.width = '100%';
                progressText.textContent = 'Complete!';

                // Show viewers
                setTimeout(() => {
                    progressContainer.classList.remove('active');
                    viewersContainer.style.display = 'grid';
                    statsPanel.style.display = 'block';

                    // Set up original viewer
                    document.getElementById('originalViewer').innerHTML = `
                        <model-viewer src="${result.original_path}"
                            alt="Original model" auto-rotate camera-controls
                            shadow-intensity="1" environment-image="neutral">
                        </model-viewer>
                    `;
                    document.getElementById('downloadOriginal').href = result.original_path;

                    // Set up primitive viewer
                    if (result.primitive_path) {
                        document.getElementById('primitiveViewer').innerHTML = `
                            <model-viewer src="${result.primitive_path}"
                                alt="Primitive model" auto-rotate camera-controls
                                shadow-intensity="1" environment-image="neutral">
                            </model-viewer>
                        `;
                        document.getElementById('downloadPrimitive').href = result.primitive_path;

                        // Update stats
                        document.getElementById('statPrimitives').textContent = result.stats.total;
                        document.getElementById('statCubes').textContent = result.stats.cubes;
                        document.getElementById('statSpheres').textContent = result.stats.spheres;
                        document.getElementById('statCylinders').textContent = result.stats.cylinders;
                        document.getElementById('statTime').textContent = result.stats.time.toFixed(1);

                        // Show Genesis code
                        if (result.genesis_code) {
                            genesisCode.classList.add('active');
                            document.getElementById('codeBlock').innerHTML = result.genesis_code;
                        }
                    } else {
                        document.getElementById('primitiveViewer').innerHTML = `
                            <div class="viewer-placeholder">
                                Model conversion requires GPU.<br>
                                Running in demo mode.
                            </div>
                        `;
                    }

                }, 500);

            } catch (err) {
                progressText.textContent = `Error: ${err.message}`;
                progressFill.style.width = '0%';
                progressFill.style.background = '#ff5252';
            }
        }
    </script>
</body>
</html>
"""


def generate_genesis_code(primitives_json):
    """Generate Genesis physics engine code from primitives."""
    if not primitives_json or 'group' not in primitives_json:
        return None

    lines = [
        '<span class="keyword">import</span> genesis <span class="keyword">as</span> gs',
        '',
        'gs.init(backend=gs.cuda)',
        '',
        'scene = gs.Scene(show_viewer=<span class="keyword">True</span>)',
        'scene.add_entity(gs.morphs.Plane())',
        '',
        '<span class="comment"># Add primitives from PrimitiveAnything</span>',
    ]

    type_map = {
        1101002001034001: ('Box', 'size'),      # CubeBevel
        1101002001034010: ('Sphere', 'radius'), # SphereSharp
        1101002001034002: ('Cylinder', 'radius, height'),  # CylinderSharp
    }

    for i, block in enumerate(primitives_json['group']):
        type_id = block['type_id']
        data = block['data']

        shape_type, params = type_map.get(type_id, ('Box', 'size'))
        pos = data['location']
        scale = data['scale']

        if shape_type == 'Box':
            lines.append(
                f'scene.add_entity(gs.morphs.Box('
                f'pos=(<span class="number">{pos[0]:.3f}</span>, <span class="number">{pos[1]:.3f}</span>, <span class="number">{pos[2]:.3f}</span>), '
                f'size=(<span class="number">{scale[0]:.3f}</span>, <span class="number">{scale[1]:.3f}</span>, <span class="number">{scale[2]:.3f}</span>)))'
            )
        elif shape_type == 'Sphere':
            avg_radius = sum(scale) / 3
            lines.append(
                f'scene.add_entity(gs.morphs.Sphere('
                f'pos=(<span class="number">{pos[0]:.3f}</span>, <span class="number">{pos[1]:.3f}</span>, <span class="number">{pos[2]:.3f}</span>), '
                f'radius=<span class="number">{avg_radius:.3f}</span>))'
            )
        elif shape_type == 'Cylinder':
            radius = (scale[0] + scale[2]) / 2
            height = scale[1]
            lines.append(
                f'scene.add_entity(gs.morphs.Cylinder('
                f'pos=(<span class="number">{pos[0]:.3f}</span>, <span class="number">{pos[1]:.3f}</span>, <span class="number">{pos[2]:.3f}</span>), '
                f'radius=<span class="number">{radius:.3f}</span>, height=<span class="number">{height:.3f}</span>))'
            )

    lines.extend([
        '',
        'scene.build()',
        '',
        '<span class="keyword">for</span> i <span class="keyword">in</span> range(<span class="number">1000</span>):',
        '    scene.step()',
    ])

    return '<br>'.join(lines)


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, model_loaded=MODEL_LOADED)


@app.route('/api/convert', methods=['POST'])
def api_convert():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'})

    file = request.files['file']
    if not file.filename:
        return jsonify({'success': False, 'error': 'No file selected'})

    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"

        task_dir = OUTPUT_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Save original file
        original_ext = Path(file.filename).suffix.lower()
        original_path = task_dir / f"original{original_ext}"
        file.save(str(original_path))

        result = {
            'success': True,
            'original_path': f'/files/{task_id}/original.glb',
            'primitive_path': None,
            'stats': {
                'total': 0,
                'cubes': 0,
                'spheres': 0,
                'cylinders': 0,
                'time': 0
            },
            'genesis_code': None
        }

        # Run conversion if model is loaded
        # run_inference will save the processed mesh as original.glb (in same coordinate space as primitives)
        if MODEL_LOADED:
            start_time = time.time()

            primitive_glb, primitives_json = run_inference(str(original_path), str(task_dir))

            elapsed = time.time() - start_time

            if primitive_glb:
                result['primitive_path'] = f'/files/{task_id}/{Path(primitive_glb).name}'

                # Count primitives by type
                cubes = spheres = cylinders = 0
                if primitives_json and 'group' in primitives_json:
                    for block in primitives_json['group']:
                        type_id = block['type_id']
                        if type_id == 1101002001034001:
                            cubes += 1
                        elif type_id == 1101002001034010:
                            spheres += 1
                        elif type_id == 1101002001034002:
                            cylinders += 1

                result['stats'] = {
                    'total': cubes + spheres + cylinders,
                    'cubes': cubes,
                    'spheres': spheres,
                    'cylinders': cylinders,
                    'time': elapsed
                }

                result['genesis_code'] = generate_genesis_code(primitives_json)
        else:
            # Model not loaded - just convert to GLB for viewing
            import trimesh
            mesh = trimesh.load(str(original_path), force='mesh')
            original_glb = task_dir / "original.glb"
            mesh.export(str(original_glb))

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


def run_inference(input_path, output_dir):
    """Run PrimitiveAnything inference on input mesh."""
    global transformer, accelerator, mesh_bs

    print(f"[DEBUG] run_inference called with {input_path}")

    if not MODEL_LOADED:
        print("[DEBUG] Model not loaded!")
        return None, None

    import torch
    import trimesh
    import numpy as np
    import seaborn as sns
    from scipy.spatial.transform import Rotation
    from accelerate.utils import set_seed
    print("[DEBUG] Imports done")

    # On macOS, mesh_to_sdf causes crashes with AppKit thread issues in Flask
    # Always use trimesh sampling on Mac for stability
    import platform
    if platform.system() == 'Darwin':
        USE_MESH_TO_SDF = False
    else:
        try:
            from mesh_to_sdf import get_surface_point_cloud
            USE_MESH_TO_SDF = True
        except Exception:
            USE_MESH_TO_SDF = False

    # Try to use mesh2sdf for watertight conversion, otherwise use trimesh voxelization
    try:
        import mesh2sdf.core
        import skimage.measure
        USE_MESH2SDF = True
    except Exception:
        USE_MESH2SDF = False

    device = DEVICE

    def normalize_vertices(vertices, scale=0.9):
        bbmin, bbmax = vertices.min(0), vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale_factor = 2.0 * scale / (bbmax - bbmin).max()
        vertices = (vertices - center) * scale_factor
        return vertices, center, scale_factor

    def export_to_watertight(mesh, octree_depth=7):
        if USE_MESH2SDF:
            # High quality watertight conversion using mesh2sdf + marching cubes
            import skimage.measure
            size = 2 ** octree_depth
            level = 2 / size
            scaled_vertices, to_orig_center, to_orig_scale = normalize_vertices(mesh.vertices)
            sdf = mesh2sdf.core.compute(scaled_vertices, mesh.faces, size=size)
            vertices, faces, normals, _ = skimage.measure.marching_cubes(np.abs(sdf), level)
            vertices = vertices / size * 2 - 1
            vertices = vertices / to_orig_scale + to_orig_center
            return trimesh.Trimesh(vertices, faces, normals=normals)
        else:
            # Fallback: use trimesh voxelization for watertight conversion
            # This is simpler but works on Mac without mesh2sdf
            scaled_vertices, to_orig_center, to_orig_scale = normalize_vertices(mesh.vertices)
            mesh_copy = mesh.copy()
            mesh_copy.vertices = scaled_vertices
            # Voxelize and convert back to mesh
            pitch = 2.0 / (2 ** octree_depth)
            voxels = mesh_copy.voxelized(pitch=pitch)
            watertight = voxels.marching_cubes
            # Restore original scale
            watertight.vertices = watertight.vertices / to_orig_scale + to_orig_center
            return watertight

    def euler_to_quat(euler):
        return Rotation.from_euler('XYZ', euler, degrees=True).as_quat()

    def SRT_quat_to_matrix(scale, quat, translation):
        rotation_matrix = Rotation.from_quat(quat).as_matrix()
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix * scale
        transform_matrix[:3, 3] = translation
        return transform_matrix

    set_seed(0)
    print("[DEBUG] set_seed done")

    # Load and preprocess mesh
    print(f"[DEBUG] Loading mesh from {input_path}")
    input_mesh = trimesh.load(input_path, force='mesh')
    print(f"[DEBUG] Mesh loaded: {len(input_mesh.vertices)} vertices, {len(input_mesh.faces)} faces")
    vertices = input_mesh.vertices
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max() * 1.6
    input_mesh.vertices = vertices
    print("[DEBUG] Mesh normalized")

    # Convert to watertight
    print("[DEBUG] Converting to watertight...")
    mesh = export_to_watertight(input_mesh)
    print(f"[DEBUG] Watertight mesh: {len(mesh.vertices)} vertices")

    # Dilate
    dilated_offset = 0.015
    new_vertices = mesh.vertices + mesh.vertex_normals * dilated_offset
    mesh.vertices = new_vertices

    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.fix_normals()

    # Sample surface points
    rng = np.random.default_rng()

    if USE_MESH_TO_SDF:
        # Use mesh_to_sdf for higher quality sampling (requires OpenGL)
        surface_point_cloud = get_surface_point_cloud(
            mesh, 'scan', None, 100, 400, 10000000, calculate_normals=True
        )
        indices = rng.choice(surface_point_cloud.points.shape[0], 10000, replace=True)
        points = surface_point_cloud.points[indices]
        normals = surface_point_cloud.normals[indices]
    else:
        # Fallback: use trimesh's built-in sampling (works on Mac without OpenGL)
        points, face_indices = trimesh.sample.sample_surface(mesh, 10000)
        normals = mesh.face_normals[face_indices]

    # Rescale mesh and points together (important: use same bounds for both)
    # This matches the demo.py flow when dilated_offset > 0
    vertices = mesh.vertices
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    center = (bounds[0] + bounds[1]) / 2
    scale = (bounds[1] - bounds[0]).max()

    # Rescale vertices
    vertices = vertices - center[None, :]
    vertices = vertices / scale * 1.6
    mesh.vertices = vertices

    # Rescale points with same transform
    points = points - center[None, :]
    points = points / scale * 1.6

    pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)
    input_pc = torch.tensor(pc_normal, dtype=torch.float16, device=device)[None]

    # Run inference - use generate() instead of generate_w_recon_loss() to avoid pytorch3d dependency
    with torch.no_grad():
        if accelerator is not None:
            with accelerator.autocast():
                recon_primitives, mask = transformer.generate(
                    pc=input_pc, temperature=0.0
                )
        else:
            # MPS or CPU - no autocast needed
            recon_primitives, mask = transformer.generate(
                pc=input_pc.float(), temperature=0.0
            )

    # Write output
    out_json = {'operation': 0, 'type': 1, 'scene_id': None, 'group': []}
    model_scene = trimesh.Scene()

    primitives = recon_primitives
    num_primitives = (primitives['type_code'].squeeze() != -1).sum().item()
    color_map = sns.color_palette("hls", num_primitives)
    color_map = (np.array(color_map) * 255).astype("uint8")

    for idx, (scale, rotation, translation, type_code) in enumerate(zip(
        primitives['scale'].squeeze().cpu().numpy(),
        primitives['rotation'].squeeze().cpu().numpy(),
        primitives['translation'].squeeze().cpu().numpy(),
        primitives['type_code'].squeeze().cpu().numpy()
    )):
        if type_code == -1:
            break

        bs_name = CODE_SHAPE[type_code]

        new_block = {
            'type_id': SHAPENAME_MAP[bs_name],
            'data': {
                'location': translation.tolist(),
                'rotation': euler_to_quat(rotation).tolist(),
                'scale': scale.tolist(),
                'color': ['808080']
            }
        }
        out_json['group'].append(new_block)

        trans = SRT_quat_to_matrix(scale, euler_to_quat(rotation), translation)
        bs = mesh_bs[bs_name].copy().apply_transform(trans)
        new_vertex_colors = np.repeat(color_map[idx:idx+1], bs.visual.vertex_colors.shape[0], axis=0)
        bs.visual.vertex_colors[:, :3] = new_vertex_colors

        # Coordinate transform
        vertices = bs.vertices.copy()
        vertices[:, 1] = bs.vertices[:, 2]
        vertices[:, 2] = -bs.vertices[:, 1]
        bs.vertices = vertices
        model_scene.add_geometry(bs)

    # Save outputs
    json_path = os.path.join(output_dir, 'primitives.json')
    with open(json_path, 'w') as f:
        json.dump(out_json, f, indent=4)

    glb_path = os.path.join(output_dir, 'primitives.glb')
    model_scene.export(glb_path)

    # Also save the processed input mesh (in same coordinate space as primitives)
    # Apply same Y/Z coordinate swap for GLB export
    processed_mesh = input_mesh.copy()
    vertices = processed_mesh.vertices.copy()
    vertices[:, 1] = processed_mesh.vertices[:, 2]
    vertices[:, 2] = -processed_mesh.vertices[:, 1]
    processed_mesh.vertices = vertices
    processed_glb_path = os.path.join(output_dir, 'original.glb')
    processed_mesh.export(processed_glb_path)

    return glb_path, out_json


@app.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)


def main():
    # Try to load model
    try_load_model()

    print(f"\n🔷 PrimitiveAnything Web App")
    print(f"   Output directory: {OUTPUT_DIR.absolute()}")
    print(f"   Model loaded: {MODEL_LOADED}")
    print(f"   Open http://localhost:5002 in your browser\n")

    app.run(host='0.0.0.0', port=5002, debug=False)


if __name__ == '__main__':
    main()
