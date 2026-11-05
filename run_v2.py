import argparse
from channel_operator.v2_data import prepare
from channel_operator.v2_train import train


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='artifacts/v2');p.add_argument('--stage',choices=['all','data','train'],default='all')
    p.add_argument('--train-cases',type=int,default=128);p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--log-every',type=int,default=250);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--points',type=int,default=96);p.add_argument('--width',type=int,default=64)
    p.add_argument('--latent',type=int,default=32);p.add_argument('--depth',type=int,default=3);p.add_argument('--resume')
    a=p.parse_args()
    if min(a.train_cases,a.steps,a.log_every,a.batch,a.points)<1:p.error('Counts must be positive')
    if a.stage in ['all','data']:prepare(a.output,a.train_cases)
    if a.stage in ['all','train']:train(a.output,a.steps,a.log_every,a.batch,a.points,a.width,a.latent,a.depth,a.resume)


if __name__=='__main__':main()
