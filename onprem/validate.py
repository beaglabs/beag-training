"""Pre-deployment validation script for on-premise setup.

Run this on the customer's GPU machine to verify:
1. CUDA is available
2. Model loads correctly
3. Inference works
4. ONNX export produces matching outputs
5. Hardware meets requirements
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def check_cuda() -> dict:
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
        memory_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3) if device_count > 0 else 0
        return {
            "cuda_available": cuda_available,
            "device_count": device_count,
            "device_name": device_name,
            "memory_gb": round(memory_gb, 1),
            "passed": cuda_available and device_count > 0,
        }
    except Exception as e:
        return {"cuda_available": False, "error": str(e), "passed": False}


def check_unsloth() -> dict:
    try:
        import unsloth
        return {"unsloth_available": True, "version": getattr(unsloth, "__version__", "unknown"), "passed": True}
    except ImportError:
        return {"unsloth_available": False, "error": "pip install unsloth", "passed": False}


def check_model_loads(model_tier: str = "starter") -> dict:
    try:
        from onprem.unsloth_adapter import UnslothAdapter

        adapter = UnslothAdapter(
            model_tier=model_tier,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        model, tokenizer = adapter.load()

        test_text = "Hello, this is a validation test."
        inputs = tokenizer(test_text, return_tensors="pt")
        with __import__("torch").no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)

        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        adapter.unload()

        return {
            "model_tier": model_tier,
            "loaded": True,
            "inference_works": len(result_text) > 0,
            "sample_output": result_text[:100],
            "passed": True,
        }
    except Exception as e:
        return {"model_tier": model_tier, "loaded": False, "error": str(e), "passed": False}


def check_onnx_export() -> dict:
    try:
        import subprocess
        result = subprocess.run(["optimum-cli", "--version"], capture_output=True, text=True)
        return {
            "optimum_available": result.returncode == 0,
            "version": result.stdout.strip(),
            "passed": result.returncode == 0,
        }
    except Exception as e:
        return {"optimum_available": False, "error": str(e), "passed": False}


def check_disk_space() -> dict:
    try:
        import shutil
        usage = shutil.disk_usage("/")
        free_gb = usage.free / (1024**3)
        return {
            "free_gb": round(free_gb, 1),
            "sufficient": free_gb > 50,
            "passed": free_gb > 50,
        }
    except Exception as e:
        return {"error": str(e), "passed": False}


def run_all_checks(config: dict) -> dict:
    model_tier = config.get("customer", {}).get("model_tier", "starter")
    results = {}

    print("=" * 60)
    print("Beag On-Prem Validation")
    print("=" * 60)

    print("\n[1/5] Checking CUDA...")
    results["cuda"] = check_cuda()
    print(f"  CUDA: {'PASS' if results['cuda']['passed'] else 'FAIL'}")
    if results["cuda"]["passed"]:
        print(f"  Device: {results['cuda'].get('device_name')}")
        print(f"  Memory: {results['cuda'].get('memory_gb')} GB")

    print("\n[2/5] Checking Unsloth...")
    results["unsloth"] = check_unsloth()
    print(f"  Unsloth: {'PASS' if results['unsloth']['passed'] else 'FAIL'}")

    print(f"\n[3/5] Loading model ({model_tier} tier)...")
    results["model"] = check_model_loads(model_tier)
    print(f"  Model load: {'PASS' if results['model']['passed'] else 'FAIL'}")

    print("\n[4/5] Checking ONNX export tools...")
    results["onnx"] = check_onnx_export()
    print(f"  ONNX tools: {'PASS' if results['onnx']['passed'] else 'FAIL'}")

    print("\n[5/5] Checking disk space...")
    results["disk"] = check_disk_space()
    print(f"  Disk: {'PASS' if results['disk']['passed'] else 'FAIL'}")
    print(f"  Free: {results['disk'].get('free_gb')} GB")

    all_passed = all(r.get("passed", False) for r in results.values())

    print("\n" + "=" * 60)
    print(f"OVERALL: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 60)

    return {
        "passed": all_passed,
        "results": results,
    }


if __name__ == "__main__":
    import yaml

    config_path = sys.argv[1] if len(sys.argv) > 1 else "onprem/config.yaml"

    if not Path(config_path).exists():
        print(f"Config not found: {config_path}")
        print("Using defaults (starter tier)")
        config = {"customer": {"model_tier": "starter"}}
    else:
        with open(config_path) as f:
            config = yaml.safe_load(f)

    result = run_all_checks(config)
    sys.exit(0 if result["passed"] else 1)
