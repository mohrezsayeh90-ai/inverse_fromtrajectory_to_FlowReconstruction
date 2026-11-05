"""Evaluate saved candidates after training; objective is never called here."""
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root=Path('output');scores=[]
    for name in ['checkpoint','best_checkpoint']:
        cp=root/(name+'.npz')
        if not cp.exists():continue
        directory=root/('validation_'+name)
        subprocess.run([sys.executable,'-m','flow_operator.evaluate','--checkpoint',str(cp),
            '--references','validation/references','--output',str(directory)],check=True)
        report=json.loads((directory/'metrics.json').read_text())
        thresholds=report['thresholds']
        score=max(c[k]/thresholds[k] for c in report['cases'] for k in thresholds)
        scores.append((score,cp,report))
    if not scores:raise RuntimeError('No completed checkpoint to evaluate')
    _,chosen,validation=min(scores,key=lambda row:row[0])
    selected=root/'selected_checkpoint.npz';shutil.copy2(chosen,selected)
    combined=dict(validation);combined['validation_role']='repeated_validation_used_for_selection'
    combined['selected_from']=chosen.name
    if validation['all_gates_passed'] and Path('validation/heldout_references').exists():
        directory=root/'heldout_test'
        subprocess.run([sys.executable,'-m','flow_operator.evaluate','--checkpoint',str(selected),
            '--references','validation/heldout_references','--output',str(directory),'--role','final_heldout_test'],check=True)
        test=json.loads((directory/'metrics.json').read_text())
        combined['heldout_test']=test
        combined['all_gates_passed']=bool(validation['all_gates_passed'] and test['all_gates_passed'])
        combined['final_test_evaluated']=True
    else:
        combined['final_test_evaluated']=False
    selected.with_suffix('.validation.json').write_text(json.dumps(combined,indent=2))
    status=dict(training_completed=True,benchmark_accepted=combined['all_gates_passed'],
        final_test_evaluated=combined['final_test_evaluated'],
        ready_after_both_suites=combined['all_gates_passed'] and combined['final_test_evaluated'],
        checkpoint=selected.name,checkpoint_sha256=combined['checkpoint_sha256'],
        universal_accuracy_guaranteed=False,CFD_training_labels=0)
    (root/'acceptance_status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
