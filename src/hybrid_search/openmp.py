import importlib.util
import os
import sys
from pathlib import Path


def _package_dir(name):
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def _paths():
    faiss_dir, torch_dir = _package_dir("faiss"), _package_dir("torch")
    if faiss_dir is None or torch_dir is None:
        return None, None
    return faiss_dir / ".dylibs" / "libomp.dylib", torch_dir / "lib" / "libomp.dylib"


def conflict():
    if sys.platform != "darwin":
        return False
    ours, theirs = _paths()
    if ours is None or not ours.exists() or not theirs.exists():
        return False
    return ours.resolve() != theirs.resolve()


def ensure_single_runtime():
    if conflict() and not os.environ.get("HSE_SKIP_OPENMP_CHECK"):
        raise RuntimeError(
            "faiss and torch each ship their own libomp on macOS; loading both "
            "segfaults inside faiss search. Run `hse fix-openmp` once to point "
            "faiss at torch's copy."
        )


def fix():
    if not conflict():
        return False
    ours, theirs = _paths()
    backup = ours.with_name("libomp.dylib.orig")
    if not backup.exists():
        ours.rename(backup)
    else:
        ours.unlink()
    ours.symlink_to(os.path.relpath(theirs, ours.parent))
    return True
