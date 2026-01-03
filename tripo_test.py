#!/usr/bin/env python3
"""
Tripo3D API Test Script

Tests the Tripo3D text-to-3D API connection and model generation.

Usage:
    1. Set your API key: export TRIPO_API_KEY="your_api_key"
    2. Run: python tripo_test.py "a wooden chair"

Get your API key at: https://platform.tripo3d.ai/
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path

try:
    from tripo3d import TripoClient
except ImportError:
    print("Error: tripo3d package not installed.")
    print("Install with: pip install tripo3d")
    sys.exit(1)


async def test_connection(client: TripoClient) -> bool:
    """Test API connection by checking balance."""
    try:
        balance = await client.get_balance()
        print(f"✓ Connected to Tripo3D API")
        print(f"  Account balance: {balance.balance}")
        print(f"  Frozen: {balance.frozen}")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


async def generate_3d_model(
    client: TripoClient,
    prompt: str,
    output_dir: str = "./tripo_output",
    timeout: float = 300.0,
) -> dict:
    """
    Generate a 3D model from text prompt.

    Args:
        client: TripoClient instance
        prompt: Text description of the 3D model
        output_dir: Directory to save output files
        timeout: Maximum time to wait for generation (seconds)

    Returns:
        Dictionary with paths to downloaded files
    """
    print(f"\n→ Generating 3D model from prompt: '{prompt}'")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Start text-to-3D generation
    print("  Submitting task...")
    task_id = await client.text_to_model(
        prompt=prompt,
        model_version="v2.5-20250123",
        texture=True,
        pbr=True,
        texture_quality="standard",
    )
    print(f"  Task ID: {task_id}")

    # Poll for completion
    print("  Waiting for generation (this may take 1-3 minutes)...")
    task = await client.wait_for_task(
        task_id,
        polling_interval=3.0,
        timeout=timeout,
        verbose=True,
    )

    if task.status != "success":
        print(f"✗ Generation failed with status: {task.status}")
        return {}

    print(f"✓ Generation complete!")

    # Download the model files
    print("  Downloading model files...")
    files = await client.download_task_models(task, output_dir)

    # Also try to download rendered image if available
    try:
        image_path = await client.download_rendered_image(task, output_dir)
        if image_path:
            files["rendered_image"] = image_path
    except Exception:
        pass  # Rendered image is optional

    # Print results
    print(f"\n✓ Downloaded files:")
    for file_type, file_path in files.items():
        if file_path:
            print(f"  - {file_type}: {file_path}")

    return files


async def main():
    parser = argparse.ArgumentParser(
        description="Test Tripo3D API for text-to-3D generation"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="a simple wooden chair",
        help="Text prompt for 3D model generation",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./tripo_output",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="Tripo API key (or set TRIPO_API_KEY env var)",
    )
    parser.add_argument(
        "--test-only", "-t",
        action="store_true",
        help="Only test connection, don't generate",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout for generation in seconds",
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("TRIPO_API_KEY")
    if not api_key:
        print("Error: No API key provided.")
        print("Set TRIPO_API_KEY environment variable or use --api-key flag")
        print("Get your API key at: https://platform.tripo3d.ai/")
        sys.exit(1)

    # Initialize client
    async with TripoClient(api_key=api_key) as client:
        # Test connection
        connected = await test_connection(client)
        if not connected:
            sys.exit(1)

        if args.test_only:
            print("\n✓ Connection test passed!")
            return

        # Generate 3D model
        files = await generate_3d_model(
            client,
            prompt=args.prompt,
            output_dir=args.output_dir,
            timeout=args.timeout,
        )

        if files:
            print(f"\n✓ Success! Model saved to: {args.output_dir}")

            # Hint about using with PrimitiveAnything
            glb_file = files.get("model") or files.get("base_model")
            if glb_file:
                print(f"\nTo convert to primitives with PrimitiveAnything:")
                print(f"  python demo.py --input {glb_file}")
        else:
            print("\n✗ Failed to generate model")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
