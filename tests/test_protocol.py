"""Protocol assembly and multi-sample summaries, without a GPU."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_paper_launcher_commands(self):
        for kind in ['main','ablation']:
            result=subprocess.run([sys.executable,str(ROOT/'reproduction/run_paper.py'),
                '--kind',kind,'--model','model','--data','data','--output','runs','--dry-run'],
                check=True,capture_output=True,text=True)
            lines=result.stdout.splitlines()
            self.assertEqual(len(lines),15)
            self.assertEqual(sum('--samples 16' in line for line in lines),9)
            self.assertEqual(sum('--samples 4' in line for line in lines),6)
            for seed in [42,985,211]:
                self.assertEqual(sum('--seed '+str(seed)+' ' in line for line in lines),5)
            self.assertTrue(all('--max-tool-calls 15' in line for line in lines))

    def test_ablation_summary_accepts_multiple_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);source=d/'input.jsonl';traces=d/'output.jsonl';report=d/'summary.json'
            source.write_text(json.dumps({'id':'q','problem':'example'})+'\n')
            cfg={'input':str(source),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                 'samples':2,'limit':0}
            traces.with_suffix('.config.json').write_text(json.dumps(cfg))
            rows=[]
            for sample in [0,1]:
                for arm in ['full','no_latest','no_delta','values_only']:
                    rows.append({'id':'q','sample':sample,'action':arm,'prefix':'p','problem':'example',
                                 'gt_answer':'1','seed':42+sample,'gen':'same','hint_tokens':0,'events':[],
                                 'correct':sample==0,'tokens':10,'calls':0})
            traces.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            subprocess.run([sys.executable,str(ROOT/'reproduction/summarize_ablation.py'),
                            '--input',str(traces),'--output',str(report)],check=True,capture_output=True)
            data=json.loads(report.read_text())
            self.assertTrue(all(m['n']==2 and m['accuracy_pct']==50 for m in data['metrics'].values()))


if __name__=='__main__':unittest.main()
