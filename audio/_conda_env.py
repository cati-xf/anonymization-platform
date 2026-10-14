import os
import shutil

_cache: dict[str, str] = {}


def _conda_root() -> str | None:
    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda_exe:
        return None
    return os.path.dirname(os.path.dirname(conda_exe))  # .../bin/conda -> conda root


def get_conda_env_bin(env_name: str, exe: str) -> str:
    """Best-effort path to `exe` inside a conda env's bin/, falling back to
    whatever `exe` resolves to on PATH. Never raises, so it is safe to call at
    import time; a wrong path only fails when the executable is actually run."""
    root = _conda_root()
    if root:
        candidate = os.path.join(root, "envs", env_name, "bin", exe)
        if os.path.exists(candidate):
            return candidate
    return shutil.which(exe) or exe


def get_conda_env_python(env_name: str, override: str | None = None) -> str:
    """Resolve a conda env's python interpreter by absolute path.

    `conda run -n <env> python` resolves `python` via PATH, which can pick up a
    different interpreter (e.g. an activated venv's) ahead of the target env's
    own bin/ directory. Invoking the env's python directly sidesteps that.
    """
    if override:
        return override
    if env_name in _cache:
        return _cache[env_name]

    conda_root = _conda_root()
    if not conda_root:
        raise RuntimeError(
            "Could not locate the 'conda' executable to resolve the "
            f"'{env_name}' env's python interpreter. Set an explicit override "
            "env var to its absolute path directly."
        )
    python_path = os.path.join(conda_root, "envs", env_name, "bin", "python")
    if not os.path.exists(python_path):
        raise RuntimeError(
            f"Expected '{env_name}' conda env python at {python_path} but it "
            "doesn't exist. Check the env name, or set an override env var to "
            "its absolute path directly."
        )
    _cache[env_name] = python_path
    return python_path
