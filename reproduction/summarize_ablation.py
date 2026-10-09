"""Validate complete multi-sample ablations and summarize one seed."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cesf.ablation_feedback import ARMS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    cfg=json.loads(a.input.with_suffix('.config.json').read_text())
    source=Path(cfg['input'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg['input_sha256']
    inputs=[json.loads(s) for s in source.read_text().splitlines()]
    if cfg['limit']:inputs=inputs[:cfg['limit']]
    expected={(r.get('id',hashlib.sha256(r['problem'].encode()).hexdigest()),sample)
              for r in inputs for sample in range(cfg['samples'])}
    groups=defaultdict(dict)
    for line in a.input.read_text().splitlines():
        r=json.loads(line);key=(r['id'],r['sample'])
        assert r['action'] not in groups[key], 'Duplicate sample/arm'
        groups[key][r['action']]=r
    assert set(groups)==expected, 'Missing or unexpected question/sample'
    for group in groups.values():
        assert set(group)==set(ARMS), 'Incomplete arms'
        full=group['full']
        for r in group.values():
            for key in ('prefix','problem','gt_answer','seed','fork_at_call','fork_tokens'):
                assert r.get(key)==full.get(key), 'Unmatched paired prefix'
            assert r['hint_tokens']==sum(e.get('state_feedback_tokens',0) for e in r['events'])
        if 'fork_at_call' not in full:
            assert len({r['gen'] for r in group.values()})==1, 'No-op trajectory mismatch'
    metrics={}
    for arm in ARMS:
        rows=[g[arm] for g in groups.values()]
        metrics[arm]={'n':len(rows),'correct':sum(r['correct'] for r in rows),
                     'accuracy_pct':100*statistics.mean(r['correct'] for r in rows),
                     'avg_charged_tokens':statistics.mean(r['tokens']+r['hint_tokens'] for r in rows),
                     'avg_feedback_tokens':statistics.mean(r['hint_tokens'] for r in rows),
                     'avg_tool_calls':statistics.mean(r['calls'] for r in rows)}
    result={'config':cfg,'metrics':metrics,
            'scope':'Per-seed sample-averaged accuracy. Aggregate the three seed reports with run_paper.py.'}
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
