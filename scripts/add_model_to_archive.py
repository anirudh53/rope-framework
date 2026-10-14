"""
add_model_to_archive.py -- put model directories into a release archive (.tar.gz or .zip) and make the
first one the default in config/rope.conf. Works for any platform's archive (model files are
platform-independent), so the CI-built macOS / Windows archives can ship with models like the Linux ones.

    python scripts/add_model_to_archive.py <archive> <out_archive> <model_dir> [model_dir ...]
"""
import os
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile


def main():
    src, dst, models = sys.argv[1], sys.argv[2], sys.argv[3:]
    if not models:
        sys.exit("need at least one model_dir")
    with tempfile.TemporaryDirectory() as tmp:
        if src.endswith(".zip"):
            with zipfile.ZipFile(src) as z:
                z.extractall(tmp)
        else:
            with tarfile.open(src) as t:
                t.extractall(tmp, filter="data")
        (top,) = [d for d in os.listdir(tmp) if os.path.isdir(os.path.join(tmp, d))]
        root = os.path.join(tmp, top)

        dest = os.path.join(root, "data", "models")
        os.makedirs(dest, exist_ok=True)
        for m in models:
            shutil.copytree(m, os.path.join(dest, os.path.basename(os.path.normpath(m))), dirs_exist_ok=True)

        conf = os.path.join(root, "config", "rope.conf")
        text = open(conf, encoding="utf-8").read()
        default = os.path.basename(os.path.normpath(models[0]))
        text, n = re.subn(r"(?m)^exported_dir\s*=.*$", f"exported_dir = ../data/models/{default}", text)
        if n != 1:
            sys.exit(f"{conf}: expected one exported_dir line, found {n}")
        open(conf, "w", encoding="utf-8", newline="").write(text)

        if dst.endswith(".zip"):
            with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
                for d, _, files in os.walk(root):
                    for f in files:
                        p = os.path.join(d, f)
                        z.write(p, os.path.relpath(p, tmp))
        else:
            with tarfile.open(dst, "w:gz") as t:
                t.add(root, arcname=top)
    print("wrote", dst)


if __name__ == "__main__":
    main()
