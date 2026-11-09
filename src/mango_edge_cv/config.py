"""
Global configuration, constants, random seed, and dataset label resolvers for
Mango-Disease-Audit-EdgeAI (`mango_edge_cv`).
"""

import os
import re
from pathlib import Path

SEED = 20261022
IMG_SIZE = 256
FEATURE_GRID_SIZE = 128
HOG_BACKEND = "numpy_sobel_vectorized_shared_gradient"
MANIFEST_SCHEMA_VERSION = "manifest-v6-mango-mad-lut-fastfg-blockcoral"
FEATURE_CACHE_VERSION = "handcrafted-v8-mango-mad-lut-fastfg-blockcoral"
CANONICAL_COLOR_MODE = "lab_l"

VALID_EXTS = {".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".webp", ".tif", ".tiff"}

MANGO_MAIN_CLASSES = [
    "Anthracnose",
    "Bacterial_Canker",
    "Cutting_Weevil",
    "Die_Back",
    "Gall_Midge",
    "Healthy",
    "Powdery_Mildew",
    "Sooty_Mold",
]

MLDID_SHARED_CLASSES = [
    "Anthracnose",
    "Bacterial_Canker",
    "Die_Back",
    "Gall_Midge",
    "Healthy",
]

FEATURE_BRANCH_NAMES = ["hsv", "lab", "lbp", "patch5", "q", "mgct", "hog"]
FEATURE_BRANCH_DIMS = {
    "hsv": 52,
    "lab": 57,
    "lbp": 28,
    "patch5": 45,
    "q": 7,
    "mgct": 50,
    "hog": 1764,
}
RAW_TOTAL_DIMS = 2003
NON_HOG_DIMS = 239
HOG_PCA_DIMS = 64
COMPRESSED_TOTAL_DIMS = 303

GENERIC_DIRS = {
    "",
    "main_data",
    "external_data",
    "original",
    "originals",
    "raw",
    "raw_data",
    "dataset",
    "data",
    "train",
    "training",
    "test",
    "testing",
    "val",
    "valid",
    "validation",
    "images",
    "image",
    "fruit",
    "stem",
    "leaf",
    "leaves",
    "archive",
    "mangoleafbd_dataset",
    "mangoleafbd",
    "resize_mango_leaf_disease",
}


def get_repo_root() -> Path:
    """Resolve the repository root directory relative to this module."""
    return Path(__file__).resolve().parents[2]


def resolve_default_paths():
    """Resolve configurable data and output directories via environment variables or defaults."""
    repo_root = get_repo_root()
    mango_root = os.environ.get(
        "MANGO_DATA_ROOT", str(repo_root / "data" / "MangoLeafBD")
    )
    mldid_root = os.environ.get(
        "MLDID_DATA_ROOT", str(repo_root / "data" / "MLDID")
    )
    output_dir = os.environ.get("OUTPUT_DIR", str(repo_root / "results"))
    return {
        "repo_root": str(repo_root),
        "mango_data_root": mango_root,
        "mldid_data_root": mldid_root,
        "output_dir": output_dir,
    }


def norm_token(value: str) -> str:
    """Normalize a folder name for rule-based matching while retaining original names in manifests."""
    value = str(value).strip().casefold()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def standardize_label(
    raw_folder_name: str,
    track_name: str = "MANGO",
    is_external: bool = False,
    rel_parts=None,
):
    """Map raw dataset directory names into canonical MangoLeafBD/MLDID disease labels."""
    raw = norm_token(raw_folder_name)
    if raw in GENERIC_DIRS:
        return None

    if track_name == "MANGO":
        if "anthracnose" in raw:
            return "Anthracnose"
        if "bacteri" in raw or "canker" in raw or "cancer" in raw:
            return "Bacterial_Canker"
        if "weevil" in raw or "cutting" in raw:
            return "Cutting_Weevil"
        if "die_back" in raw or "dieback" in raw:
            return "Die_Back"
        if "gall" in raw or "midge" in raw or "mid_disease" in raw or "gall_mid" in raw:
            return "Gall_Midge"
        if "healthy" in raw:
            return "Healthy"
        if "powdery" in raw or "mildew" in raw:
            return "Powdery_Mildew"
        if "sooty" in raw or "mold" in raw or "mould" in raw:
            return "Sooty_Mold"
        return None

    return None


def resolve_label_from_relpath(rel_dir: str, track_name: str = "MANGO", is_external: bool = False):
    """Search from the leaf folder upward and return (canonical_label, matched_folder)."""
    parts = [p for p in str(rel_dir).replace("\\", "/").split("/") if p and p != "."]
    for index in range(len(parts) - 1, -1, -1):
        label = standardize_label(
            parts[index], track_name=track_name, is_external=is_external, rel_parts=parts
        )
        if label is not None:
            return label, parts[index]
    return None, (parts[-1] if parts else "")


def scan_dataset_records(data_root: str, track_name: str = "MANGO", is_external: bool = False):
    """
    Scan a dataset directory tree and collect valid image records with canonical labels.
    Skips explicit 'Augmented' or 'Balanced' directory components.
    """
    records = []
    discovered_folders = set()
    root_abs = os.path.abspath(data_root)
    if not os.path.isdir(root_abs):
        return records, discovered_folders

    for root, dirs, files in os.walk(root_abs):
        dirs[:] = sorted(
            d
            for d in dirs
            if d.casefold() not in {"__macosx", ".ipynb_checkpoints", "__pycache__"}
        )
        rel_dir = os.path.relpath(root, root_abs)
        rel_parts = [] if rel_dir == "." else rel_dir.replace("\\", "/").split("/")
        normalized_parts = [norm_token(p) for p in rel_parts]

        if any(("augment" in part) or ("balanced" in part) for part in normalized_parts):
            dirs[:] = []
            continue

        valid_files = [
            f
            for f in sorted(files)
            if not f.startswith(".")
            and not f.startswith("._")
            and os.path.splitext(f)[1].casefold() in VALID_EXTS
        ]
        if not valid_files:
            continue

        label_name, matched_folder = resolve_label_from_relpath(
            rel_dir, track_name=track_name, is_external=is_external
        )
        discovered_folders.add(rel_dir)
        if label_name is None:
            continue

        for filename in valid_files:
            path = os.path.join(root, filename)
            try:
                size_kb = os.path.getsize(path) / 1024.0
            except OSError:
                size_kb = float("nan")
            records.append(
                {
                    "file_path": path,
                    "filename": filename,
                    "raw_folder": matched_folder,
                    "label_name": label_name,
                    "file_size_kb": size_kb,
                    "relative_folder": rel_dir.replace("\\", "/"),
                }
            )

    records.sort(key=lambda row: (row["label_name"], row["file_path"].casefold()))
    return records, discovered_folders

