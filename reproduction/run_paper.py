"""Launch the current manuscript protocol; --dry-run performs no inference."""
import argparse
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['main','ablation'],default='main')
    p.add_argument('--model',required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    here=Path(__file__).resolve().parent
    protocol=json.loads((here/'paper_protocol.json').read_text())
    runner=here/('run_paired.py' if a.kind=='main' else 'run_ablation.py')
    summarize=here/('summarize_paired.py' if a.kind=='main' else 'summarize_ablation.py')
    aggregate={}
    for dataset,setting in protocol['benchmarks'].items():
        reports=[]
        for seed in protocol['seeds']:
            source=a.data/('test_'+dataset+'.reason_step.jsonl')
            output=a.output/(dataset+'_seed'+str(seed)+'.jsonl')
            report=output.with_suffix('.report.json')
            cmd=[sys.executable,str(runner),'--model',a.model,'--input',str(source),'--output',str(output),
                 '--mode','paired','--samples',str(setting['samples']),'--seed',str(seed),
                 '--budget',str(protocol['max_tokens']),'--context',str(protocol['max_tokens']),
                 '--max-tool-calls',str(protocol['max_tool_calls']),'--batch','16' if a.kind=='main' else '8']
            summary=[sys.executable,str(summarize),'--input',str(output),'--output',str(report)]
            print(shlex.join(cmd),flush=True)
            if not a.dry_run:
                if not source.is_file(): raise FileNotFoundError(source)
                subprocess.run(cmd,check=True)
                subprocess.run(summary,check=True)
                reports.append(json.loads(report.read_text()))
        if reports:
            metrics=[{'baseline':r['control']['accuracy_pct'],'CESF':r['state_sync']['accuracy_pct']} for r in reports] if a.kind=='main' else [{k:v['accuracy_pct'] for k,v in r['metrics'].items()} for r in reports]
            aggregate[dataset]={k:{'mean_accuracy_pct':statistics.mean(m[k] for m in metrics),
                                  'seed_accuracies_pct':[m[k] for m in metrics]} for k in metrics[0]}
    if not a.dry_run:
        (a.output/'three_seed_summary.json').write_text(json.dumps({'protocol':protocol,'results':aggregate},indent=2))


if __name__=='__main__':
    main()
