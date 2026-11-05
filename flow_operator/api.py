"""Reusable frozen-weight inference with one JIT cache for successive inlets."""
import hashlib
import json
from pathlib import Path

import numpy as np
from .checkpoint import load
from .model import make_functions
from .profiles import stack
from .provenance import runtime_signature


class FlowOperator:
    def __init__(self,checkpoint,allow_unvalidated=False,chunk_size=1024):
        self.path=Path(checkpoint)
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        report_path=self.path.with_suffix('.validation.json')
        report=json.loads(report_path.read_text()) if report_path.exists() else {}
        self.benchmark_accepted=(report.get('checkpoint_sha256')==self.sha256 and report.get('all_gates_passed') is True
                                 and report.get('runtime_signature')==runtime_signature())
        if not self.benchmark_accepted and not allow_unvalidated:
            raise ValueError('No passing benchmark report for these weights; provisional use requires allow_unvalidated=True')
        if chunk_size<1:raise ValueError('chunk_size must be positive')
        self.params,self.config,self.metadata,_=load(checkpoint)
        self.functions=make_functions(self.config);self.chunk_size=chunk_size

    def evaluate(self,inlet,reynolds,xy):
        """Return dimensionless [u,v,p]; no optimization, CFD, or response lookup."""
        xy=np.asarray(xy,dtype=np.float32)
        if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():raise ValueError('xy must be finite with shape (N,2)')
        if not np.isfinite(reynolds) or reynolds<=0:raise ValueError('Reynolds must be positive and finite')
        if np.any(xy<0) or np.any(xy[:,0]>self.config.length) or np.any(xy[:,1]>1):raise ValueError('Requested coordinates are outside the channel')
        if not len(xy):return np.empty((0,3))
        data=stack([inlet]);re=np.array([reynolds],np.float32);result=[]
        for left in range(0,len(xy),self.chunk_size):
            part=xy[left:left+self.chunk_size];actual=len(part)
            padded=np.pad(part,((0,self.chunk_size-actual),(0,0)),mode='edge')
            fields=np.asarray(self.functions['fields'](self.params,data,re,padded[None]))[0,:actual]
            if not np.isfinite(fields).all():raise RuntimeError('Nonfinite prediction')
            result.append(fields)
        return np.concatenate(result)
