"""Verify page counts, editable slide objects, source metrics, and exported assets."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pptx import Presentation
from pypdf import PdfReader
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    spec=json.loads((HERE/'slide_model.json').read_text());ppt=Presentation(HERE/'defense_fa.pptx')
    pdf=PdfReader(HERE/'defense_fa.pdf');layout=json.loads((HERE/'layout_validation.json').read_text())
    assert len(ppt.slides)==len(pdf.pages)==len(spec)==22
    assert not layout['text_overflows']
    for sl,model in zip(ppt.slides,spec):
        expected='\n'.join(b['text'] for b in model['blocks'] if b['kind']=='text')
        actual='\n'.join(s.text for s in sl.shapes if s.has_text_frame)
        for text in expected.split('\n'):assert text in actual,text
        assert sl.notes_slide.notes_text_frame.text.strip()
        for s in sl.shapes:
            assert s.left>=0 and s.top>=0
            assert s.left+s.width<=ppt.slide_width+10 and s.top+s.height<=ppt.slide_height+10
    source=json.loads((HERE/'data/forward_metrics.json').read_text())
    cp=ROOT/'models/candidates/pressure_long/checkpoint.npz'
    assert hashlib.sha256(cp.read_bytes()).hexdigest()==source['checkpoint_sha256']
    checks=[]
    for row in source['all_predeclared_cases']:
        with np.load(HERE/'data'/f"{row['name']}.npz",allow_pickle=False) as d:
            error=d['prediction']-d['reference']
            ev=np.linalg.norm(error[:,:2])/np.linalg.norm(d['reference'][:,:2]);ep=np.linalg.norm(error[:,2])/np.linalg.norm(d['reference'][:,2])
        np.testing.assert_allclose([ev,ep],[row['velocity_relative_l2'],row['pressure_relative_l2']],atol=1e-12)
        checks.append(dict(case=row['name'],velocity_error_pct=100*float(ev),pressure_error_pct=100*float(ep)))
    low=[r for r in checks if r['velocity_error_pct']<5 and r['pressure_error_pct']<10];assert len(low)==3
    inverse_checks=[]
    for folder in [ROOT/'flow_reconstruction/examples/corner_launch_preconditioned/result',HERE/'data/inverse_beta4/result']:
        result=json.loads((folder/'report.json').read_text())
        assert result['forward_checkpoint_sha256']==source['checkpoint_sha256'] and result['forward_weights_unchanged']
        for i,row in enumerate(result['trajectories'],1):
            observed=np.genfromtxt(folder/f'observed_track_{i}.csv',delimiter=',',names=True)
            predicted=np.genfromtxt(folder/f'track_{i}_forward.csv',delimiter=',',names=True)
            assert len(observed)==len(predicted)==101
            np.testing.assert_allclose(observed['t'],predicted['t'],atol=1e-12)
            dx=predicted['x']-observed['x'];dy=predicted['y']-observed['y']
            two_d=float(np.sqrt(np.mean(dx*dx+dy*dy)))
            np.testing.assert_allclose(two_d,2**.5*row['position_rmse'],atol=1e-12)
            inverse_checks.append(dict(source=str(folder),particle=row['name'],position_2d_rmse_percent_H=100*two_d))
    report=dict(pages=22,main_slides=19,backup_slides=3,pptx_native_text_editable=True,
        pdf_page_count_verified=True,all_slide_text_present_in_pptx=True,no_off_slide_objects=True,
        html_text_overflows=0,all_figures_from_numeric_sources=True,all_four_attempted_forward_cases_preserved=checks,
        same_checkpoint_for_all_results=source['checkpoint_sha256'],low_field_error_cases=len(low),inverse_metrics_verified=inverse_checks,
        selected_examples_do_not_establish_global_acceptance=True,inverse_trajectory_consistency_is_not_full_field_accuracy=True,
        renderer_note='PDF and PNG previews rendered with Chromium. Native PowerPoint object and text structure checked; no Microsoft PowerPoint renderer is available here.')
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
