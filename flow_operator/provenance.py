"""Bind numerical acceptance to the code that interprets the saved parameters."""
import hashlib
from pathlib import Path


def runtime_signature():
    root=Path(__file__).resolve().parent
    h=hashlib.sha256()
    for name in ['model.py','profiles.py','checkpoint.py']:
        h.update(name.encode());h.update((root/name).read_bytes())
    return h.hexdigest()
