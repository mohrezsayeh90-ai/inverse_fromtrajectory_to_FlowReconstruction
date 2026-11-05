import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from channel_operator.v2_evaluate import evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--role',choices=['repeated_monitor_validation','untouched_final_blind_test'],required=True)
    a=p.parse_args();evaluate(a.checkpoint,a.data,a.output,a.role)
