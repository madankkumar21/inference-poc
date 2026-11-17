"""The Modal app: vLLM's OpenAI-compatible server on one GPU.

Deploy one app per variant; the variant is read from the local environment at deploy time
and baked into the image, so the container sees the same value:

    POC_VARIANT=fp8 POC_WEIGHTS=<hf id> POC_ARGS='[...]' modal deploy providers/modal_app.py   # app "inference-poc-fp8"
    modal run providers/modal_app.py::download --model <hf id>    # fill the weights Volume once (CPU only)

providers/modal_vllm.py sets these variables from config/poc.yaml; you do not set them by hand.

This file imports nothing from the repo: Modal ships it to the container on its own.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
import time
import urllib.request

import modal

VARIANT = os.environ.get("POC_VARIANT", "bf16")
SERVED_NAME = os.environ.get("POC_SERVED_NAME", "Qwen/Qwen2.5-7B-Instruct")   # what clients send
WEIGHTS = os.environ.get("POC_WEIGHTS", SERVED_NAME)                         # what vLLM loads
# Inside the container the args arrive base64-encoded: Modal's image env step drops the
# backslashes in nested JSON such as '{"enable_thinking": false}'.
if os.environ.get("POC_ARGS_B64"):
    EXTRA_ARGS = json.loads(base64.b64decode(os.environ["POC_ARGS_B64"]).decode())
else:
    EXTRA_ARGS = json.loads(os.environ.get("POC_ARGS", "[]"))
GPU = os.environ.get("POC_GPU", "L40S")
CPU = float(os.environ.get("POC_CPU", "2"))
MEMORY_MIB = int(float(os.environ.get("POC_MEMORY_GIB", "24")) * 1024)
PORT = 8000
MAX_MODEL_LEN = int(os.environ.get("POC_MAX_MODEL_LEN", "12288"))   # 8K-token prompts plus output, with headroom
VLLM_VERSION = os.environ.get("POC_VLLM", "0.11.0")
# The original arms pin transformers and huggingface_hub; newer vLLM releases resolve their own.
TRANSFORMERS = os.environ.get("POC_TRANSFORMERS", "4.57.1" if VLLM_VERSION == "0.11.0" else "")
PINS = [f"transformers=={TRANSFORMERS}", "huggingface_hub>=0.34,<1"] if TRANSFORMERS else []

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}", *PINS)
    .env({
        "POC_VARIANT": VARIANT,
        "POC_SERVED_NAME": SERVED_NAME,
        "POC_WEIGHTS": WEIGHTS,
        "POC_ARGS_B64": base64.b64encode(json.dumps(EXTRA_ARGS).encode()).decode(),
        "POC_GPU": GPU,
        "POC_VLLM": VLLM_VERSION,
        "POC_TRANSFORMERS": TRANSFORMERS,
        "POC_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
        "HF_HOME": "/cache/hf",
        "VLLM_LOGGING_LEVEL": "INFO",
    })
)

app = modal.App(f"inference-poc-{VARIANT}")
hf_vol = modal.Volume.from_name("inference-poc-hf", create_if_missing=True)
compile_vol = modal.Volume.from_name("inference-poc-vllm-cache", create_if_missing=True)
timings = modal.Dict.from_name("inference-poc-timings", create_if_missing=True)


@app.function(image=image, volumes={"/cache/hf": hf_vol}, timeout=60 * 60)
def download(model: str = WEIGHTS):
    """Fetch weights into the Volume so cold starts measure loading, not the Hub download."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(model, ignore_patterns=["*.pt", "*.bin", "*.pth"])
    hf_vol.commit()
    print(f"weights at {path}")


@app.function(
    image=image,
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MIB,
    volumes={"/cache/hf": hf_vol, "/root/.cache/vllm": compile_vol},
    secrets=[modal.Secret.from_name("inference-poc-key")],
    scaledown_window=120,   # scale to zero two minutes after the last request
    timeout=60 * 60,
    max_containers=1,       # one GPU, so the benchmark measures one replica
)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def serve():
    t_container = time.time()
    cmd = [
        "vllm", "serve", WEIGHTS,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--api-key", os.environ["VLLM_API_KEY"],
        *EXTRA_ARGS,
    ]
    print("starting:", " ".join(c if c != os.environ["VLLM_API_KEY"] else "***" for c in cmd))
    subprocess.Popen(cmd)

    def record_ready():
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(0.25)
        t_ready = time.time()
        gpu_name = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip()
        rec = {
            "variant": VARIANT, "weights": WEIGHTS, "args": EXTRA_ARGS, "gpu": gpu_name or GPU, "vllm": VLLM_VERSION,
            "container_start": t_container, "server_ready": t_ready,
        }
        timings[f"{VARIANT}:{int(t_container * 1000)}"] = rec
        print("ready:", json.dumps(rec))
        compile_vol.commit()   # keep torch.compile artifacts for the next cold start

    threading.Thread(target=record_ready, daemon=True).start()
