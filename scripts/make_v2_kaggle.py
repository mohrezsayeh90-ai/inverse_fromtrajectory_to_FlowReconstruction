"""Package tested sources and synthetic FEM arrays into a private GPU notebook."""
import argparse
import base64
import hashlib
import json
import uuid
import io
import numpy as np
from pathlib import Path


def code(s):
    return dict(cell_type='code',id=uuid.uuid4().hex[:8],metadata={},execution_count=None,outputs=[],source=s.splitlines(keepends=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--id',default='mohrezsayeh/channel-stream-deeponet-v2')
    a=p.parse_args();root=Path(__file__).resolve().parents[1];data=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    files=sorted(root.glob('channel_operator/*.py'))+[root/'run_v2.py',root/'tests/test_v2_physics.py']
    sources={str(f.relative_to(root)):f.read_text() for f in files}
    # Keep the notebook below Kaggle's code-payload limits. Preserve all monitor
    # points; compact only the training anchors. Physics points remain online.
    asset_bytes={}
    packaging={'dtype':'float32','monitor_points_reduced':False,'original_train_points':None,'packed_train_points':256}
    for name in ['train','monitor']:
        with np.load(data/f'{name}.npz') as d: arrays={k:d[k] for k in d.files}
        if name=='train':
            packaging['original_train_points']=len(arrays['xy'])
            rng=np.random.default_rng(6732)
            near=np.arange(min(75,len(arrays['xy'])))
            rest=rng.choice(np.arange(len(near),len(arrays['xy'])),256-len(near),replace=False)
            selected=np.sort(np.concatenate([near,rest]))
            arrays['xy']=arrays['xy'][selected];arrays['fields']=arrays['fields'][:,selected]
            packaging['selected_train_point_indices']=selected.tolist()
        buffer=io.BytesIO();np.savez_compressed(buffer,**{k:v.astype(np.float32) for k,v in arrays.items()})
        asset_bytes[f'{name}.npz']=buffer.getvalue()
    asset_bytes['reference_metadata.json']=(data/'reference_metadata.json').read_bytes()
    asset_bytes['packaging.json']=json.dumps(packaging,indent=2).encode()
    assets={name:base64.b64encode(content).decode() for name,content in asset_bytes.items()}
    manifest={name:hashlib.sha256(text.encode()).hexdigest() for name,text in sources.items()}
    manifest['assets']={name:hashlib.sha256(content).hexdigest() for name,content in asset_bytes.items()}
    setup='import os,sys,subprocess,json,base64\nfrom pathlib import Path\nos.environ["MPLCONFIGDIR"]="/kaggle/working/.mpl"\nos.environ["OMP_NUM_THREADS"]="4"\n'
    setup+='subprocess.run([sys.executable,"-m","pip","install","scikit-fem==12.0.2","pytest>=8,<10"],check=True)\n'
    setup+='import torch\nprint("CUDA",torch.cuda.is_available(),"PyTorch",torch.__version__)\nassert torch.cuda.is_available()\nprint("GPU",torch.cuda.get_device_name(0))\n'
    setup+='sources='+repr(sources)+'\nfor name,text in sources.items():\n    path=Path(name)\n    path.parent.mkdir(parents=True,exist_ok=True)\n    path.write_text(text)\n'
    setup+='assets='+repr(assets)+'\nPath("artifacts").mkdir(exist_ok=True)\nfor name,text in assets.items():\n    (Path("artifacts")/name).write_bytes(base64.b64decode(text))\n'
    setup+='Path("artifacts/source_manifest.json").write_text('+repr(json.dumps(manifest,indent=2))+')\n'
    setup+='subprocess.run([sys.executable,"-m","pytest","-q","tests/test_v2_physics.py"],check=True)\n'
    run='subprocess.run([sys.executable,"run_v2.py","--stage","train","--output","artifacts","--steps",'+repr(str(a.steps))+',"--log-every","250"],check=True)\n'
    report='print(Path("artifacts/v2_training_summary.json").read_text())\n'
    notebook=dict(cells=[dict(cell_type='markdown',id='intro-v2',metadata={},source=['# Streamfunction physics-informed DeepONet v2\n','Unseen severe FEM conditions are monitoring-only. Every training print includes CFD errors.\n']),code(setup),code(run),code(report)],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},nbformat=4,nbformat_minor=5)
    (out/'stream_v2.ipynb').write_text(json.dumps(notebook))
    (out/'kernel-metadata.json').write_text(json.dumps({'id':a.id,'title':a.id.split('/')[-1].replace('-',' ').title(),'code_file':'stream_v2.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'dataset_sources':[],'competition_sources':[],'kernel_sources':[]},indent=2))
    print('Notebook bytes', (out/'stream_v2.ipynb').stat().st_size)


if __name__=='__main__':main()
