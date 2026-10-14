#!/usr/bin/env bash
# Run ONCE on the machine where the platform already works, from the repo root,
# with the main venv activated. Records the exact versions used so the README's
# install steps are reproducible. Commit the files it writes.
set -u
cd "$(dirname "$0")/.."

pip freeze --exclude-editable > requirements.lock.txt

for env in f5-tts aligner diarizen; do
    conda env export -n "$env" --from-history 2>/dev/null | grep -v '^prefix:' > "envs/$env.yml" \
        || echo "!! conda env '$env' not found"
    # Call the env's own python directly: `conda run -n X pip` resolves pip via
    # PATH and can pick up the activated venv's pip instead. Envs installed purely
    # through conda (e.g. aligner/MFA) have no pip, so record conda packages instead.
    env_python="$(conda info --base)/envs/$env/bin/python"
    if "$env_python" -m pip --version >/dev/null 2>&1; then
        "$env_python" -m pip freeze --exclude-editable > "envs/$env.pip-freeze.txt"
    else
        conda list -n "$env" --export > "envs/$env.conda-list.txt" 2>/dev/null \
            || echo "!! could not record packages for '$env'"
    fi
done

{
    echo "# Environment snapshot -- $(date -I)"
    echo "OS:      $(. /etc/os-release && echo "$PRETTY_NAME")"
    echo "Python:  $(python --version 2>&1)"
    echo "Conda:   $(conda --version 2>&1)"
    echo "CUDA:    $(nvcc --version 2>/dev/null | tail -1)"
    echo "Driver:  $(nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader 2>/dev/null | sort -u)"
    echo "ffmpeg:  $(ffmpeg -version 2>/dev/null | head -1)"
    echo "espeak:  $(espeak-ng --version 2>/dev/null)"
    echo "torch:   $(python -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>&1)"
    echo
    echo "F5-TTS:  $(git -C "${F5_TTS_DIR:-$HOME/F5-TTS/F5-TTS}" remote get-url origin 2>/dev/null) @ $(git -C "${F5_TTS_DIR:-$HOME/F5-TTS/F5-TTS}" rev-parse HEAD 2>/dev/null)"
    for d in "${TUSS_DIR:-$HOME/standalone_inference}" "$HOME/DiariZen"; do
        [ -d "$d/.git" ] && echo "$(basename "$d"): $(git -C "$d" remote get-url origin) @ $(git -C "$d" rev-parse HEAD)"
    done
    echo
    echo "MFA models:"; conda run -n aligner mfa model list acoustic 2>/dev/null; conda run -n aligner mfa model list dictionary 2>/dev/null
    echo "Ollama models:"; ollama list 2>/dev/null
} > envs/SNAPSHOT.txt

echo "Wrote requirements.lock.txt and envs/ -- check them, then commit."
