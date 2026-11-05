"""Private compiled run reusing the immutable CFD-output source on Kaggle."""
import argparse,json,hashlib,uuid
from pathlib import Path


def cell(s):return dict(cell_type='code',id=uuid.uuid4().hex[:8],metadata={},execution_count=None,outputs=[],source=s.splitlines(keepends=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=50000)
    p.add_argument('--id',default='mohrezsayeh/channel-stream-jax-v2')
    p.add_argument('--pressure-mode',choices=['viscous','convective'],default='viscous');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    files=sorted(root.glob('channel_operator/*.py'))+[root/'run_v2_jax.py',root/'tests/test_v2_jax_parity.py',root/'tests/test_v2_physics.py']
    sources={str(f.relative_to(root)):f.read_text() for f in files}
    manifest={name:hashlib.sha256(text.encode()).hexdigest() for name,text in sources.items()}
    setup='import os,sys,subprocess,json\nfrom pathlib import Path\nos.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"\nos.environ["MPLCONFIGDIR"]="/kaggle/working/.mpl"\nos.environ["OMP_NUM_THREADS"]="4"\n'
    setup+='import jax\nprint("JAX",jax.__version__,"DEVICES",jax.devices())\nassert any(d.platform=="gpu" for d in jax.devices())\n'
    setup+='sources='+repr(sources)+'\nfor name,text in sources.items():\n    path=Path(name)\n    path.parent.mkdir(parents=True,exist_ok=True)\n    path.write_text(text)\n'
    setup+='candidates=[p.parent for p in Path("/kaggle/input").rglob("train.npz") if (p.parent/"monitor.npz").exists()]\nassert len(candidates)==1, candidates\nDATA_DIR=candidates[0]\nprint("REFERENCE_INPUT",DATA_DIR)\n'
    setup+='Path("artifacts").mkdir(exist_ok=True)\nPath("artifacts/source_manifest.json").write_text('+repr(json.dumps(manifest,indent=2))+')\n'
    setup+='subprocess.run([sys.executable,"-m","pytest","-q","tests/test_v2_jax_parity.py","tests/test_v2_physics.py"],check=True)\n'
    run='subprocess.run([sys.executable,"run_v2_jax.py","--data",str(DATA_DIR),"--output","artifacts","--steps",'+repr(str(a.steps))+',"--batch","16","--points","96","--log-every","250","--require-gpu"],check=True)\n'
    if a.pressure_mode!='viscous':run=run.replace('],check=True)',',"--pressure-mode",'+repr(a.pressure_mode)+'],check=True)')
    notebook=dict(cells=[dict(cell_type='markdown',id='jax-v2-intro',metadata={},source=['# Compiled physics-informed streamfunction DeepONet\n','Numerically checked against PyTorch, including third derivatives and parameter gradients. Monitoring CFD responses are never used by the optimizer.\n']),cell(setup),cell(run),cell('print(Path("artifacts/v2_training_summary.json").read_text())\n')],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},nbformat=4,nbformat_minor=5)
    (out/'stream_jax.ipynb').write_text(json.dumps(notebook))
    (out/'kernel-metadata.json').write_text(json.dumps({'id':a.id,'title':a.id.split('/')[-1].replace('-',' ').title(),'code_file':'stream_jax.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,'dataset_sources':[],'competition_sources':[],'kernel_sources':['mohrezsayeh/channel-v2-fem-reference']},indent=2))
    print('Notebook bytes',(out/'stream_jax.ipynb').stat().st_size)


if __name__=='__main__':main()
