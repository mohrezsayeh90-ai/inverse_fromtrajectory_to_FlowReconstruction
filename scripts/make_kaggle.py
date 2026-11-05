"""Package only NEW operator sources, no old weights and no training responses."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def code_cell(text):return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(keepends=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='kaggle_run');p.add_argument('--id',default='mohrezsayeh/channel-general-physics-v3-probe')
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--resume-source');p.add_argument('--include-references',action='store_true')
    p.add_argument('--lift',choices=['gaussian','oseen'],default='oseen')
    p.add_argument('--physics-normalization',choices=['none','input_curvature','input_curvature_sqrt'],default='input_curvature_sqrt')
    p.add_argument('--functional-moments',action='store_true')
    p.add_argument('--encoder-kind',choices=['pooled','sensor'],default='sensor')
    p.add_argument('--batch',type=int,default=64);p.add_argument('--points',type=int,default=48)
    p.add_argument('--width',type=int,default=128);p.add_argument('--latent',type=int,default=64)
    p.add_argument('--depth',type=int,default=4)
    p.add_argument('--separate-pressure-trunk',action='store_true')
    p.add_argument('--devices',type=int,default=1)
    p.add_argument('--final-evaluation',action='store_true')
    p.add_argument('--include-heldout',action='store_true')
    p.add_argument('--reuse-validation-from-resume',action='store_true')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.final_evaluation and not a.include_references:p.error('final-evaluation requires include-references')
    if a.functional_moments and a.encoder_kind!='pooled':p.error('functional-moments requires --encoder-kind pooled')
    if a.reuse_validation_from_resume and not (a.resume_source and a.include_references):
        p.error('reuse-validation-from-resume requires resume-source and include-references')
    files=sorted((ROOT/'flow_operator').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))+[ROOT/'pyproject.toml',ROOT/'configs/acceptance.json']
    if a.devices>1:files+=[ROOT/'scripts/verify_parallel.py']
    if a.final_evaluation:files+=[ROOT/'scripts/finalize_run.py']
    sources={str(path.relative_to(ROOT)):path.read_text() for path in files}
    refs={};inherited_refs={}
    if a.include_references:
        refs={str(path.relative_to(ROOT)):base64.b64encode(path.read_bytes()).decode() for path in (ROOT/'validation/references').glob('*.npz')}
        if len(refs)<4:raise ValueError('Expected the four independent validation references')
        if a.reuse_validation_from_resume:
            inherited_refs={name:hashlib.sha256(base64.b64decode(value)).hexdigest() for name,value in refs.items()}
            refs={}
    if a.include_heldout:
        if not a.final_evaluation or not a.include_references:p.error('Heldout requires final-evaluation and validation references')
        refs.update({str(path.relative_to(ROOT)):base64.b64encode(path.read_bytes()).decode() for path in (ROOT/'validation/heldout_references').glob('*.npz')})
    manifest={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
    setup='''import os,sys,json,base64,subprocess
from pathlib import Path
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['MPLCONFIGDIR']='/kaggle/working/.mpl'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import jax
print('JAX',jax.__version__,'DEVICES',jax.devices())
assert any(d.platform=='gpu' for d in jax.devices()), 'GPU required'
'''
    setup+='sources='+repr(sources)+'\n'
    setup+='for name,text in sources.items():\n    path=Path(name)\n    path.parent.mkdir(parents=True,exist_ok=True)\n    path.write_text(text)\n'
    setup+='reference_files='+repr(refs)+'\n'
    setup+='for name,text in reference_files.items():\n    path=Path(name)\n    path.parent.mkdir(parents=True,exist_ok=True)\n    path.write_bytes(base64.b64decode(text))\n'
    if inherited_refs:
        setup+='import hashlib\ninherited_references='+repr(inherited_refs)+'\n'
        setup+='for name,digest in inherited_references.items():\n    suffix=Path(name).parts\n    matches=[p for p in Path("/kaggle/input").rglob(Path(name).name) if '+repr(a.resume_source.split('/')[-1])+' in p.parts and p.parts[-len(suffix):]==suffix]\n    assert len(matches)==1,(name,matches)\n    value=matches[0].read_bytes()\n    assert hashlib.sha256(value).hexdigest()==digest, "Inherited reference differs: "+name\n    path=Path(name)\n    path.parent.mkdir(parents=True,exist_ok=True)\n    path.write_bytes(value)\n'
    setup+='Path("output").mkdir(exist_ok=True)\nPath("output/source_manifest.json").write_text('+repr(json.dumps(manifest,indent=2))+')\n'
    reference_manifest={name:dict(sha256=hashlib.sha256(base64.b64decode(value)).hexdigest(),origin='embedded') for name,value in refs.items()}
    reference_manifest.update({name:dict(sha256=digest,origin=a.resume_source) for name,digest in inherited_refs.items()})
    setup+='Path("output/reference_manifest.json").write_text('+repr(json.dumps(reference_manifest,indent=2))+')\n'
    setup+='subprocess.run([sys.executable,"-m","pytest","-q"],check=True)\n'
    if a.devices>1:setup+='subprocess.run([sys.executable,"-m","scripts.verify_parallel"],check=True)\n'
    command=['-m','flow_operator.train','--output','output','--steps',str(a.steps),'--batch',str(a.batch),'--points',str(a.points),
             '--pool-count','10000','--log-every','250','--require-gpu']
    command+=['--width',str(a.width),'--latent',str(a.latent),'--depth',str(a.depth)]
    command+=['--lift',a.lift,'--physics-normalization',a.physics_normalization]
    if a.functional_moments:command+=['--functional-moments']
    command+=['--encoder-kind',a.encoder_kind]
    if a.separate_pressure_trunk:command+=['--separate-pressure-trunk']
    command+=['--devices',str(a.devices)]
    if refs or inherited_refs:command+=['--reference-monitor','validation/references','--reference-every','1000']
    run='command='+repr(command)+'\n'
    if a.resume_source:
        run+='matches=[p for p in Path("/kaggle/input").rglob("checkpoint.npz") if '+repr(a.resume_source.split('/')[-1])+' in p.parts]\nassert len(matches)==1,matches\ncommand += ["--resume",str(matches[0])]\n'
    run+='subprocess.run([sys.executable,*command],check=True)\n'
    if a.final_evaluation:run+='subprocess.run([sys.executable,"-m","scripts.finalize_run"],check=True)\n'
    notebook=dict(cells=[dict(cell_type='markdown',metadata={},source=[
        '# Full-profile physics-only channel operator\n',
        'Complete supplied spline in hard inlet boundary; variable-quadrature function encoder.\n',
        'No CFD response labels in the objective. Benchmark references, if present, are read-only monitors.\n',
        'Candidate run: successful execution alone does not certify prediction accuracy.\n']),code_cell(setup),code_cell(run),
        code_cell('print(Path("output/summary.json").read_text())\n')],
        metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},nbformat=4,nbformat_minor=5)
    # The API joins cell sources before enforcing its notebook-source limit.
    api_notebook=json.loads(json.dumps(notebook))
    for cell in api_notebook['cells']:cell['source']=''.join(cell['source'])
    if len(json.dumps(api_notebook).encode())>=1024*1024:
        raise ValueError('Kaggle source exceeds 1 MB; reuse verified validation references from a completed resume source with --reuse-validation-from-resume')
    (out/'operator.ipynb').write_text(json.dumps(notebook))
    (out/'kernel-metadata.json').write_text(json.dumps(dict(id=a.id,title=a.id.split('/')[-1].replace('-',' ').title(),
        code_file='operator.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=False,
        dataset_sources=[],competition_sources=[],kernel_sources=[a.resume_source] if a.resume_source else []),indent=2))
    (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Packaged',len(sources),'new source files,',len(refs),'embedded and',len(inherited_refs),'hash-verified inherited reference files')


if __name__=='__main__':main()
