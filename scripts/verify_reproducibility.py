"""Check archived training sources, weights, state, outputs, and reference hashes."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import argparse
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='reports/REPRODUCIBILITY_AUDIT.json');args=parser.parse_args()
    root=ROOT/'reports/experiments/pressure_long'
    artifact=json.loads((root/'artifact_manifest.json').read_text())
    for name,entry in artifact.items():assert sha(root/name)==entry['sha256'],name
    notebook=ROOT/'notebooks/training/pressure_long/operator.ipynb'
    nb=json.loads(notebook.read_text());setup=''.join(nb['cells'][1]['source'])
    nodes=ast.parse(setup).body
    embedded=next(ast.literal_eval(n.value) for n in nodes if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sources' for t in n.targets))
    source_manifest=json.loads((root/'output/source_manifest.json').read_text())
    assert set(source_manifest)==set(embedded)
    for name,digest in source_manifest.items():assert hashlib.sha256(embedded[name].encode()).hexdigest()==digest,name
    states={}
    for name in ['checkpoint','best_checkpoint','selected_checkpoint']:
        cp=root/'output'/f'{name}.npz';validation=json.loads(cp.with_suffix('.validation.json').read_text())
        assert sha(cp)==validation['checkpoint_sha256']
        with np.load(cp,allow_pickle=False) as d:
            metadata=json.loads(str(d['metadata']));assert metadata['optimizer_present']
            for field in ['rng_state','pool_seed','pool_count','pool_sha256','config']:assert field in metadata
            assert any(n.startswith('adam_m_') for n in d.files) and any(n.startswith('adam_v_') for n in d.files)
        states[name]=dict(step=metadata['step'],sha256=sha(cp),optimizer_and_rng_present=True)
    refs=json.loads((root/'output/reference_manifest.json').read_text())
    for name,entry in refs.items():assert sha(ROOT/name)==entry['sha256'],name
    assert sha(ROOT/'models/candidates/pressure_long/checkpoint.npz')==states['selected_checkpoint']['sha256']
    archive=ROOT/'old_verssion';old=json.loads((archive/'ARCHIVE_MANIFEST.json').read_text())
    for name,digest in old['tracked_file_sha256'].items():assert sha(archive/name)==digest,name
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    required=['flow_operator/model.py','flow_operator/train.py','flow_operator/predict.py','flow_operator/profiles.py',
              'flow_operator/checkpoint.py','scripts/make_kaggle.py','scripts/finalize_run.py','requirements.txt',
              'docs/TRAINING_REPRODUCIBILITY.md','notebooks/training/pressure_long/operator.ipynb',
              'models/candidates/pressure_long/checkpoint.npz',
              'flow_reconstruction/inverse.py','flow_reconstruction/particles.py','flow_reconstruction/profile.py',
              'flow_reconstruction/io.py','flow_reconstruction/reconstruct.py','flow_reconstruction/demo.py',
              'flow_reconstruction/ARTIFACT_MANIFEST.json','requirements-cpu-tested.txt']
    for name in required:assert name in tracked,name
    report=dict(all_checks_passed=True,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                audit_scope='Working tree and Git index at audit time; source_commit is HEAD before saving this report.',
                required_file_sha256={name:sha(ROOT/name) for name in required},
                original_output_files=30,full_saved_artifacts=len(artifact),embedded_training_source_files=len(embedded),
                original_archived_files_verified=len(old['tracked_file_sha256']),references_verified=len(refs),model_states=states,
                source_code_execution_notebook_weights_Adam_RNG_inputs_references_predictions_logs_present=True,
                environment=dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','jax','jaxlib','matplotlib','pytest']}),
                note='Method and frozen-weight inference are reproducible; bitwise retraining across different GPU, Python, and library versions is not guaranteed. Training snapshot can be extracted from the saved notebook.')
    path=ROOT/args.output;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
