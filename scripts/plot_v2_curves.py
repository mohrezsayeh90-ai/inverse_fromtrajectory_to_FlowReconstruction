import argparse,json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--curves',nargs='+',required=True,help='label=path pairs')
    p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    rows=[];case_rows=[]
    for item in a.curves:
        label,path=item.split('=',1);records={}
        for line in Path(path).read_text().splitlines():
            try:m=json.loads(line)
            except ValueError:continue
            if 'step' in m and 'validation'in m:records.setdefault(m['step'],{}).update(m)
        records=[records[k] for k in sorted(records)]
        for m in records:
            val=m['validation'];phy=m.get('validation_physics',{})
            rows.append({'experiment':label,'step':m['step'],'seconds':m.get('seconds'),'velocity_max_percent':100*val['velocity_rel_l2_max'],
                'pressure_max_percent':100*val['p_rel_l2_max'],'velocity_mean_percent':100*val['velocity_rel_l2_mean'],
                'pressure_mean_percent':100*val['p_rel_l2_mean'],'momentum_rms_max':phy.get('momentum_rms_max'),'outlet_rms_max':phy.get('outlet_rms_max'),
                'data_weight':m.get('data_weight'),'loss':m.get('loss')})
            for c in val['cases']:
                case_rows.append({'experiment':label,'step':m['step'],**c})
        steps=[m['step'] for m in records]
        axes[0,0].semilogy(steps,[100*m['validation']['velocity_rel_l2_max'] for m in records],label=label)
        axes[0,1].semilogy(steps,[100*m['validation']['p_rel_l2_max'] for m in records],label=label)
        diagnostic=[m for m in records if 'validation_physics'in m]
        axes[1,0].semilogy([m['step'] for m in diagnostic],[m['validation_physics']['momentum_rms_max'] for m in diagnostic],label=label)
        axes[1,1].semilogy([m['step'] for m in diagnostic],[m['validation_physics']['outlet_rms_max'] for m in diagnostic],label=label)
    for ax,title,threshold in zip(axes.ravel(),['Unseen severe velocity error, max (%)','Unseen severe pressure error, max (%)','Unseen NS momentum RMS, max','Unseen outlet RMS, max'],[5,10,.05,.1]):
        ax.axhline(threshold,color='black',linestyle='--',alpha=.6);ax.set_title(title);ax.set_xlabel('optimizer step');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'learning_curves.png',dpi=160);fig.savefig(out/'learning_curves.svg');plt.close(fig)
    for filename,items in [('iteration_errors.csv',rows),('per_case_errors.csv',case_rows)]:
        keys=list(items[0]);
        with open(out/filename,'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');writer.writeheader();writer.writerows(items)
    print('Summary rows',len(rows),'per-case rows',len(case_rows))


if __name__=='__main__':main()
