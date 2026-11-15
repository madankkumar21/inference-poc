#!/usr/bin/env bash
# One-time Modal setup, after `uv run modal setup` has linked this machine to your Modal account.
# 1. writes a random endpoint key to .env (POC_VLLM_KEY) if none is there
# 2. stores the same key as the Modal secret "inference-poc-key" (VLLM_API_KEY), which vLLM checks
# 3. downloads every variant's weights into the Volume (CPU-only function, no GPU time),
#    so cold starts measure loading, not the Hugging Face download
set -euo pipefail
cd "$(dirname "$0")/.."

touch .env
if ! grep -q '^POC_VLLM_KEY=.' .env; then
  sed -i '' '/^POC_VLLM_KEY=/d' .env 2>/dev/null || true
  echo "POC_VLLM_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
  echo "wrote POC_VLLM_KEY to .env"
fi
KEY=$(grep '^POC_VLLM_KEY=' .env | cut -d= -f2-)

uv run modal secret create --force inference-poc-key "VLLM_API_KEY=$KEY" >/dev/null
echo "Modal secret inference-poc-key is set"

for W in $(uv run python -c "import yaml; v=yaml.safe_load(open('config/poc.yaml'))['providers']['modal_vllm']['variants']; print(' '.join(sorted({x['weights'] for x in v.values()})))"); do
  echo "downloading $W"
  uv run modal run providers/modal_app.py::download --model "$W"
done
uv run modal volume ls inference-poc-hf
