"""
Automated environment setup: clone AISFormer, download weights and KITTI data.

Usage:
    python setup_environment.py
"""
import os
import sys
import subprocess
import shutil
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))
from src.config import (
    AISFORMER_DIR, AISFORMER_WEIGHTS, AISFORMER_REPO_URL,
    AISFORMER_WEIGHTS_GDRIVE_FOLDER_ID, KITTI_TRACK_DIR, KITTI_IMAGES_URL,
    KITTI_LABELS_URL, DATA_DIR, IMG_BASE, LBL_BASE
)


def download_file(url, dest, description="file"):
    """Download a file using urllib (no wget dependency)."""
    print(f"  Downloading {description}...")
    try:
        urllib.request.urlretrieve(url, dest, _download_progress)
        print()  # newline after progress
    except Exception as e:
        print(f"\n  ERROR: Download failed: {e}")
        print(f"  You can manually download from:\n    {url}")
        print(f"  and place it at:\n    {dest}")
        return False
    return True


def _download_progress(block_num, block_size, total_size):
    """Progress callback for urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)", end="", flush=True)
    else:
        mb = downloaded / 1e6
        print(f"\r  {mb:.0f} MB downloaded", end="", flush=True)


def clone_aisformer():
    """Clone and build AISFormer repository."""
    if AISFORMER_DIR.exists():
        print("AISFormer already cloned.")
        return

    print("Cloning AISFormer repository...")
    subprocess.run(
        ["git", "clone", AISFORMER_REPO_URL, str(AISFORMER_DIR)],
        check=True
    )

    # Patch builtin.py for Python 3.9+ compatibility
    patch_aisformer_builtin()

    print("Building AISFormer (this may take a few minutes)...")
    subprocess.run(
        [sys.executable, "setup.py", "build", "develop"],
        cwd=str(AISFORMER_DIR),
        check=True
    )
    print("AISFormer built successfully.")


def patch_aisformer_builtin():
    """Patch AISFormer's builtin.py for Python 3.9+ compatibility.

    The original code uses 'from collections import Container' which was
    removed in Python 3.10. This patches it to use collections.abc instead.
    """
    builtin_path = AISFORMER_DIR / "detectron2" / "config" / "compat.py"
    if not builtin_path.exists():
        # Try alternative path
        builtin_path = AISFORMER_DIR / "adet" / "config.py"

    # Also check the main detectron2 config defaults
    for search_dir in [AISFORMER_DIR]:
        for root, dirs, files in os.walk(search_dir):
            for fname in files:
                if not fname.endswith('.py'):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if 'from collections import' in content and 'collections.abc' not in content:
                        # Check for deprecated imports
                        old = 'from collections import Container'
                        new = 'from collections.abc import Container'
                        if old in content:
                            content = content.replace(old, new)
                            with open(fpath, 'w', encoding='utf-8') as f:
                                f.write(content)
                            print(f"  Patched: {os.path.relpath(fpath, str(AISFORMER_DIR))}")
                except Exception:
                    pass

    print("  AISFormer compatibility patches applied.")


def download_weights():
    """Download AISFormer KINS-retrained weights from public Google Drive folder."""
    if AISFORMER_WEIGHTS.exists():
        print("AISFormer weights already present.")
        return

    os.makedirs(str(AISFORMER_WEIGHTS.parent), exist_ok=True)

    print("Downloading AISFormer KINS weights (~413 MB)...")
    # Public folder: https://drive.google.com/drive/folders/1NJhpPlbtkNBSukhT4tRPZhqHbGmvcwLI

    # Method 1: gdown folder download
    try:
        import gdown
        import glob
        tmp_dir = str(AISFORMER_WEIGHTS.parent / "_download_tmp")
        gdown.download_folder(
            id=AISFORMER_WEIGHTS_GDRIVE_FOLDER_ID,
            output=tmp_dir,
            quiet=False,
            remaining_ok=True,
        )
        candidates = glob.glob(os.path.join(tmp_dir, "**", "model_final.pth"), recursive=True)
        if candidates:
            shutil.copy(candidates[0], str(AISFORMER_WEIGHTS))
            print(f"Downloaded model_final.pth (640K iterations)")
        shutil.rmtree(tmp_dir, ignore_errors=True)

        if AISFORMER_WEIGHTS.exists() and AISFORMER_WEIGHTS.stat().st_size > 1e6:
            size_mb = AISFORMER_WEIGHTS.stat().st_size / 1e6
            print(f"Weights ready: {size_mb:.0f} MB")
            return
    except ImportError:
        print("  gdown not installed. Install with: pip install gdown")
    except Exception as e:
        print(f"  gdown failed: {e}")

    if not AISFORMER_WEIGHTS.exists():
        print("\nERROR: Could not download weights automatically.")
        print("Please download manually:")
        print(f"  1. Go to: https://drive.google.com/drive/folders/{AISFORMER_WEIGHTS_GDRIVE_FOLDER_ID}")
        print(f"  2. Download 'model_final.pth'")
        print(f"  3. Place it at: {AISFORMER_WEIGHTS}")


def download_kitti():
    """Download KITTI Tracking dataset."""
    os.makedirs(str(KITTI_TRACK_DIR), exist_ok=True)

    img_dir = KITTI_TRACK_DIR / "training" / "image_02"
    lbl_dir = KITTI_TRACK_DIR / "training" / "label_02"

    if not img_dir.exists():
        print("Downloading KITTI Tracking images (~12 GB)...")
        print("This may take 10-20 minutes depending on connection speed.")
        zip_path = str(KITTI_TRACK_DIR / "data_tracking_image_2.zip")

        success = download_file(KITTI_IMAGES_URL, zip_path, "KITTI images")
        if success:
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(str(KITTI_TRACK_DIR))
            os.remove(zip_path)
            print("Images extracted.")
    else:
        print("KITTI images already present.")

    if not lbl_dir.exists():
        print("Downloading KITTI Tracking labels...")
        zip_path = str(KITTI_TRACK_DIR / "data_tracking_label_2.zip")

        success = download_file(KITTI_LABELS_URL, zip_path, "KITTI labels")
        if success:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(str(KITTI_TRACK_DIR))
            os.remove(zip_path)
            print("Labels extracted.")
    else:
        print("KITTI labels already present.")


def verify_setup():
    """Verify all components are ready."""
    print("\n" + "=" * 50)
    print("SETUP VERIFICATION")
    print("=" * 50)

    checks = [
        ("AISFormer repo", AISFORMER_DIR.exists()),
        ("AISFormer weights", AISFORMER_WEIGHTS.exists()),
        ("KITTI images", (KITTI_TRACK_DIR / "training" / "image_02").exists()),
        ("KITTI labels", (KITTI_TRACK_DIR / "training" / "label_02").exists()),
    ]

    all_ok = True
    for name, ok in checks:
        status = "OK" if ok else "MISSING"
        print(f"  {name:25s} [{status}]")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll components ready. You can now run: python run_pipeline.py")
    else:
        print("\nSome components are missing. Please check the errors above.")

    return all_ok


def main():
    print("=" * 50)
    print("ENVIRONMENT SETUP")
    print("=" * 50)

    clone_aisformer()
    download_weights()
    download_kitti()
    verify_setup()


if __name__ == "__main__":
    main()
