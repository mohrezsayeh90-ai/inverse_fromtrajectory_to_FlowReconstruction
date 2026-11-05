import argparse,json,uuid
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    files=[root/'channel_operator'/name for name in ['__init__.py','profiles.py','v2_profiles.py','reference.py']]+[root/'scripts/augment_v2_doe.py']
    sources={str(f.relative_to(root)):f.read_text() for f in files}
    source='import subprocess,sys\nfrom pathlib import Path\nsubprocess.run([sys.executable,"-m","pip","install","scikit-fem==12.0.2"],check=True)\n'
    source+='sources='+repr(sources)+'\nfor name,text in sources.items():\n    p=Path(name)\n    p.parent.mkdir(parents=True,exist_ok=True)\n    p.write_text(text)\n'
    source+='candidates=[p.parent for p in Path("/kaggle/input").rglob("train.npz") if (p.parent/"monitor.npz").exists()]\nassert len(candidates)==1,candidates\n'
    source+='subprocess.run([sys.executable,"scripts/augment_v2_doe.py","--data",str(candidates[0]),"--output","artifacts"],check=True)\n'
    (out/'reference_doe.py').write_text(source)
    (out/'kernel-metadata.json').write_text(json.dumps({'id':'mohrezsayeh/channel-v2-doe-reference','title':'Channel V2 Doe Reference','code_file':'reference_doe.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,'enable_internet':True,'dataset_sources':[],'competition_sources':[],'kernel_sources':['mohrezsayeh/channel-v2-fem-reference']},indent=2))


if __name__=='__main__':main()
