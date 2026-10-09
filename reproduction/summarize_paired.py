import argparse,collections,hashlib,json,pathlib,statistics
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',required=True)
p.add_argument('--output',required=True);p.add_argument('--phase',default='manuscript protocol evaluation')
a=p.parse_args();path=pathlib.Path(a.input)
cfg=json.loads(path.with_suffix('.config.json').read_text())
source=pathlib.Path(cfg['input'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg['input_sha256']
inputs=[json.loads(s) for s in source.read_text().splitlines()]
if cfg['limit']: inputs=inputs[:cfg['limit']]
expected={(r.get('id',hashlib.sha256(r['problem'].encode()).hexdigest()),s)
          for r in inputs for s in range(cfg['samples'])}
groups=collections.defaultdict(dict)
for line in path.read_text().splitlines():
    r=json.loads(line);k=(r['id'],r['sample'])
    assert r['action'] not in groups[k], 'Duplicate result'
    groups[k][r['action']]=r
assert set(groups)==expected, 'Incomplete evaluation'
control=[];candidate=[];deltas=collections.defaultdict(list)
for k,g in groups.items():
    assert set(g)=={'natural','state_sync'}
    x,y=g['natural'],g['state_sync']
    for f in ('prefix','problem','gt_answer','seed'):
        assert x[f]==y[f]
    if not y['intervened']: assert x['gen']==y['gen'], 'No-op mismatch'
    if y['intervened']:
        assert x['fork_at_call']==y['fork_at_call']
        assert x['fork_tokens']==y['fork_tokens']
    assert y['hint_tokens']==sum(e.get('state_feedback_tokens',0) for e in y['events'])
    control.append(x);candidate.append(y)
    deltas[x['id']].append(float(y['correct'])-float(x['correct']))
def metrics(rs):
    return {'n':len(rs),'accuracy_pct':100*statistics.mean(r['correct'] for r in rs),
            'avg_tokens':statistics.mean(r['tokens']+r['hint_tokens'] for r in rs),
            'avg_calls':statistics.mean(r['calls'] for r in rs)}
v=np.array([np.mean(d) for d in deltas.values()]);rng=np.random.default_rng(cfg['seed'])
b=[100*rng.choice(v,len(v),replace=True).mean() for _ in range(2000)]
report={'phase':a.phase,'control':metrics(control),'state_sync':metrics(candidate),
        'delta_pp':100*v.mean(),'bootstrap_95ci_pp':np.quantile(b,[.025,.975]).tolist(),
        'interventions':sum(r['intervened'] for r in candidate),
        'wins':sum(y['correct'] and not x['correct'] for x,y in zip(control,candidate)),
        'harms':sum(x['correct'] and not y['correct'] for x,y in zip(control,candidate)),
        'config':cfg,'caveat':'Per-seed paired comparison. Aggregate all three manuscript seeds using run_paper.py; these intervals resample questions within a seed.'}
pathlib.Path(a.output).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
