#!/usr/bin/env python3
"""
Tripo3D Text-to-3D Simple Web App

A minimal Flask-based interface for generating 3D models from text using Tripo3D API.
No heavy dependencies - just Flask and the Tripo SDK.

Usage:
    1. Set your API key: export TRIPO_API_KEY="your_api_key"
    2. Run: python tripo_web.py
    3. Open http://localhost:5000 in your browser
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from functools import wraps

try:
    from flask import Flask, render_template_string, request, jsonify, send_from_directory
except ImportError:
    print("Error: flask package not installed.")
    print("Install with: pip3 install flask")
    sys.exit(1)

try:
    from tripo3d import TripoClient
except ImportError:
    print("Error: tripo3d package not installed.")
    print("Install with: pip3 install tripo3d")
    sys.exit(1)


app = Flask(__name__)
OUTPUT_DIR = Path("./tripo_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Store active tasks
active_tasks = {}


def async_route(f):
    """Decorator to run async functions in Flask routes."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tripo3D Text-to-3D Generator</title>
    <script src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js" type="module"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 {
            color: #00d4ff;
            text-align: center;
            margin-bottom: 30px;
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        @media (max-width: 768px) {
            .container { grid-template-columns: 1fr; }
        }
        .panel {
            background: #16213e;
            border-radius: 12px;
            padding: 25px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #00d4ff;
            font-weight: 500;
        }
        input, textarea, select {
            width: 100%;
            padding: 12px;
            border: 1px solid #0f3460;
            border-radius: 8px;
            background: #0f3460;
            color: #fff;
            margin-bottom: 20px;
            font-size: 14px;
        }
        textarea { resize: vertical; min-height: 100px; }
        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #00d4ff;
        }
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #00d4ff, #0077b6);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
        }
        button:disabled {
            background: #555;
            cursor: not-allowed;
        }
        .checkbox-row {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .checkbox-row label {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 0;
        }
        .checkbox-row input[type="checkbox"] {
            width: auto;
            margin: 0;
        }
        #status {
            margin-top: 20px;
            padding: 15px;
            background: #0f3460;
            border-radius: 8px;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 13px;
            max-height: 200px;
            overflow-y: auto;
        }
        model-viewer {
            width: 100%;
            height: 450px;
            background: #0a0a1a;
            border-radius: 12px;
        }
        .viewer-placeholder {
            width: 100%;
            height: 450px;
            background: #0a0a1a;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #555;
            font-size: 18px;
        }
        .balance {
            text-align: right;
            color: #00d4ff;
            margin-bottom: 20px;
        }
        .examples {
            margin-top: 20px;
        }
        .examples h3 { color: #00d4ff; margin-bottom: 10px; }
        .example-btn {
            display: inline-block;
            padding: 8px 16px;
            margin: 5px;
            background: #0f3460;
            border: 1px solid #00d4ff;
            border-radius: 20px;
            color: #00d4ff;
            cursor: pointer;
            font-size: 13px;
            transition: background 0.2s;
        }
        .example-btn:hover {
            background: #00d4ff;
            color: #000;
        }
        .download-link {
            display: inline-block;
            margin-top: 15px;
            padding: 10px 20px;
            background: #00d4ff;
            color: #000;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
        }
        .progress-bar {
            height: 4px;
            background: #0f3460;
            border-radius: 2px;
            margin-top: 10px;
            overflow: hidden;
        }
        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #0077b6);
            width: 0%;
            transition: width 0.3s;
        }
    </style>
</head>
<body>
    <h1>🎨 Tripo3D Text-to-3D Generator</h1>

    <div class="container">
        <div class="panel">
            <div class="balance" id="balance">Checking balance...</div>

            <label for="prompt">Prompt</label>
            <textarea id="prompt" placeholder="Describe the 3D model you want to create...">a simple wooden chair</textarea>

            <label for="negative">Negative Prompt (optional)</label>
            <input type="text" id="negative" placeholder="What to avoid...">

            <div class="checkbox-row">
                <label><input type="checkbox" id="texture" checked> Texture</label>
                <label><input type="checkbox" id="pbr" checked> PBR Materials</label>
            </div>

            <label for="quality">Texture Quality</label>
            <select id="quality">
                <option value="standard">Standard</option>
                <option value="detailed">Detailed</option>
            </select>

            <button id="generateBtn" onclick="generate()">Generate 3D Model</button>

            <div class="progress-bar"><div class="progress-bar-fill" id="progressBar"></div></div>

            <div id="status">Ready to generate...</div>

            <div class="examples">
                <h3>Example Prompts</h3>
                <span class="example-btn" onclick="setPrompt('a wooden dining chair with curved backrest')">wooden chair</span>
                <span class="example-btn" onclick="setPrompt('a potted succulent plant in a ceramic pot')">potted plant</span>
                <span class="example-btn" onclick="setPrompt('a medieval knight helmet with visor')">knight helmet</span>
                <span class="example-btn" onclick="setPrompt('a vintage rotary telephone')">rotary phone</span>
                <span class="example-btn" onclick="setPrompt('a cartoon treasure chest with gold')">treasure chest</span>
                <span class="example-btn" onclick="setPrompt('a cute low-poly fox')">low-poly fox</span>
            </div>
        </div>

        <div class="panel">
            <div id="viewerContainer">
                <div class="viewer-placeholder" id="placeholder">
                    Your 3D model will appear here
                </div>
            </div>
            <div id="downloadArea"></div>
        </div>
    </div>

    <script>
        let generating = false;

        // Check balance on load
        fetch('/api/balance')
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('balance').textContent =
                        `Balance: ${data.balance} | Frozen: ${data.frozen}`;
                } else {
                    document.getElementById('balance').textContent = 'Balance: Error';
                }
            });

        function setPrompt(text) {
            document.getElementById('prompt').value = text;
        }

        function updateStatus(msg, progress = null) {
            document.getElementById('status').textContent = msg;
            if (progress !== null) {
                document.getElementById('progressBar').style.width = progress + '%';
            }
        }

        async function generate() {
            if (generating) return;

            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) {
                updateStatus('Please enter a prompt');
                return;
            }

            generating = true;
            const btn = document.getElementById('generateBtn');
            btn.disabled = true;
            btn.textContent = 'Generating...';

            updateStatus('Submitting task...', 10);

            try {
                // Start generation
                const startRes = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        prompt: prompt,
                        negative_prompt: document.getElementById('negative').value,
                        texture: document.getElementById('texture').checked,
                        pbr: document.getElementById('pbr').checked,
                        texture_quality: document.getElementById('quality').value
                    })
                });

                const startData = await startRes.json();
                if (!startData.success) {
                    throw new Error(startData.error);
                }

                const taskId = startData.task_id;
                updateStatus(`Task started: ${taskId}\\nPolling for completion...`, 20);

                // Poll for completion
                let complete = false;
                while (!complete) {
                    await new Promise(r => setTimeout(r, 3000));

                    const statusRes = await fetch(`/api/status/${taskId}`);
                    const statusData = await statusRes.json();

                    if (statusData.status === 'success') {
                        complete = true;
                        updateStatus(`Generation complete!\\nDownloading model...`, 90);

                        // Load model viewer
                        const modelPath = statusData.model_path;
                        const viewerHtml = `
                            <model-viewer
                                src="${modelPath}"
                                alt="Generated 3D model"
                                auto-rotate
                                camera-controls
                                shadow-intensity="1"
                                environment-image="neutral">
                            </model-viewer>
                        `;
                        document.getElementById('viewerContainer').innerHTML = viewerHtml;
                        document.getElementById('downloadArea').innerHTML =
                            `<a href="${modelPath}" download class="download-link">Download GLB Model</a>`;

                        updateStatus(`✓ Success!\\n\\nTask ID: ${taskId}\\nModel: ${modelPath}`, 100);

                    } else if (statusData.status === 'failed' || statusData.status === 'cancelled') {
                        throw new Error(`Generation ${statusData.status}`);
                    } else {
                        const progress = statusData.progress || 50;
                        updateStatus(`Status: ${statusData.status}\\nProgress: ${progress}%`, 20 + progress * 0.7);
                    }
                }

            } catch (err) {
                updateStatus(`Error: ${err.message}`, 0);
            } finally {
                generating = false;
                btn.disabled = false;
                btn.textContent = 'Generate 3D Model';
            }
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/balance')
@async_route
async def api_balance():
    try:
        api_key = os.environ.get('TRIPO_API_KEY')
        if not api_key:
            return jsonify({'success': False, 'error': 'No API key'})

        async with TripoClient(api_key=api_key) as client:
            balance = await client.get_balance()
            return jsonify({
                'success': True,
                'balance': balance.balance,
                'frozen': balance.frozen
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/generate', methods=['POST'])
@async_route
async def api_generate():
    try:
        data = request.json
        api_key = os.environ.get('TRIPO_API_KEY')
        if not api_key:
            return jsonify({'success': False, 'error': 'No API key configured'})

        async with TripoClient(api_key=api_key) as client:
            task_id = await client.text_to_model(
                prompt=data['prompt'],
                negative_prompt=data.get('negative_prompt') or None,
                model_version="v2.5-20250123",
                texture=data.get('texture', True),
                pbr=data.get('pbr', True),
                texture_quality=data.get('texture_quality', 'standard'),
            )

            active_tasks[task_id] = {'status': 'queued', 'progress': 0}

            return jsonify({
                'success': True,
                'task_id': task_id
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/status/<task_id>')
@async_route
async def api_status(task_id):
    try:
        api_key = os.environ.get('TRIPO_API_KEY')
        async with TripoClient(api_key=api_key) as client:
            task = await client.get_task(task_id)

            if task.status == 'success':
                # Download the model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                task_output_dir = OUTPUT_DIR / f"{timestamp}_{task_id[:8]}"
                task_output_dir.mkdir(parents=True, exist_ok=True)

                files = await client.download_task_models(task, str(task_output_dir))
                model_path = files.get("model") or files.get("pbr_model") or files.get("base_model")

                if model_path:
                    return jsonify({
                        'status': 'success',
                        'model_path': f'/models/{Path(model_path).parent.name}/{Path(model_path).name}'
                    })

            return jsonify({
                'status': task.status,
                'progress': getattr(task, 'progress', 50)
            })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})


@app.route('/models/<path:filename>')
def serve_model(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == '__main__':
    api_key = os.environ.get('TRIPO_API_KEY')
    if not api_key:
        print("Warning: TRIPO_API_KEY not set!")
        print("Set it with: export TRIPO_API_KEY='your_key'")

    print(f"\n🎨 Tripo3D Text-to-3D Web App")
    print(f"   Output directory: {OUTPUT_DIR.absolute()}")
    print(f"   Open http://localhost:5001 in your browser\n")

    app.run(host='0.0.0.0', port=5001, debug=False)
