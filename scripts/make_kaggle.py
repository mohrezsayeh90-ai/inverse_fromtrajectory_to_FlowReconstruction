"""Build a self-contained, private Kaggle notebook; contains source, never credentials."""
import argparse
import hashlib
import json
import uuid
from pathlib import Path


def cell(source):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "execution_count": None, "outputs": [], "source": source.splitlines(keepends=True)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", default="kaggle-run")
    parser.add_argument("--id", default="mohrezsayeh/channel-pi-deeponet-re10-100")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--train-cases", type=int, default=48)
    parser.add_argument("--entrance-features", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = Path(args.destination)
    target.mkdir(parents=True, exist_ok=True)
    files = sorted(root.glob("channel_operator/*.py"))+sorted(root.glob("tests/*.py"))+[root/"run_experiment.py"]
    source = {str(p.relative_to(root)): p.read_text() for p in files}
    manifest = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in source.items()}
    setup = 'import os, json, subprocess, sys\nfrom pathlib import Path\nos.environ["MPLCONFIGDIR"]="/kaggle/working/.mpl"\n'
    setup += 'subprocess.run([sys.executable,"-m","pip","install","scikit-fem==12.0.2","pytest>=8,<10"],check=True)\n'
    setup += 'import torch\nprint("PyTorch:",torch.__version__,"CUDA:",torch.cuda.is_available())\nassert torch.cuda.is_available(), "GPU is required for this experiment"\nprint("GPU:",torch.cuda.get_device_name(0))\n'
    setup += 'sources = '+repr(source)+'\n'
    setup += 'for name, content in sources.items():\n    path=Path(name)\n    path.parent.mkdir(parents=True,exist_ok=True)\n    path.write_text(content)\n'
    setup += 'Path("artifacts").mkdir(exist_ok=True)\nPath("artifacts/source_manifest.json").write_text('+repr(json.dumps(manifest, indent=2))+')\n'
    setup += 'subprocess.run([sys.executable,"-m","pytest","-q"],check=True)\n'
    run = 'subprocess.run([sys.executable,"run_experiment.py","--output","artifacts","--steps",'+repr(str(args.steps))+',"--train-cases",'+repr(str(args.train_cases))+'],check=True)\n'
    if args.entrance_features:
        run = run.replace('],check=True)', ',"--entrance-features"],check=True)')
    report = 'from IPython.display import display, Image\nprint(Path("artifacts/metrics.json").read_text())\ndisplay(Image(filename="artifacts/heldout_fields.png"))\n'
    notebook = {"cells": [{"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": ["# Conditional Navier–Stokes channel operator\n", "Private GPU experiment. Hybrid FEM + physics-informed DeepONet; held-out FEM validation. No account credentials are embedded.\n"]}, cell(setup), cell(run), cell(report)],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
    (target/"channel.ipynb").write_text(json.dumps(notebook))
    (target/"kernel-metadata.json").write_text(json.dumps({"id": args.id, "title": args.id.split("/")[-1].replace("-", " ").title(), "code_file": "channel.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, indent=2))
    print(target)


if __name__ == "__main__":
    main()
