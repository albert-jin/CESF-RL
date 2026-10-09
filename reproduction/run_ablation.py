"""CESF evaluation adapted to the manuscript protocol with paired trajectories and resumable traces."""
import argparse, copy, concurrent.futures, hashlib, json, os, pathlib, subprocess, sys, time
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if os.environ.get("CORT_ROOT"):
    sys.path.insert(0,os.environ["CORT_ROOT"])
try:
    from infer.parser import extract_jupyter_like_program,extract_program
except ImportError as exc:
    raise SystemExit("Set CORT_ROOT to a local Github.CoRT checkout containing infer/parser.py.") from exc

HINT=''
ARMS = ['full', 'no_latest', 'no_delta', 'values_only']

def tool(payload):
    try:
        p=subprocess.run([sys.executable,str(ROOT/'reproduction/tool_worker_ablation.py'),'--structured-input'],input=json.dumps(payload),text=True,capture_output=True,timeout=30,
                         env={'PATH':os.environ['PATH'],'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONHASHSEED':'0'})
        return json.loads(p.stdout) if p.returncode == 0 else {'text':'Execution Error: '+p.stderr[-700:],'ok':False}
    except Exception as e:
        return {'text':'Execution Error: '+str(e),'ok':False}

def grade(answer, generation):
    from math_verify import parse,verify
    from math_verify.parser import ExprExtractionConfig,LatexExtractionConfig
    try:
        gold=parse('$'+str(answer).strip('$')+'$')
        pred=parse(generation,extraction_config=(ExprExtractionConfig(),LatexExtractionConfig()))
        return bool(verify(gold,pred))
    except Exception:
        return False

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--mode',choices=['paired','baseline','state_sync'],default='paired')
    p.add_argument('--prefix',type=int,default=512)
    p.add_argument('--budget',type=int,default=32768);p.add_argument('--context',type=int,default=32768)
    p.add_argument('--samples',type=int,required=True);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--max-tool-calls',type=int,default=15)
    args=p.parse_args()
    assert args.mode == 'paired' and args.samples >= 1
    assert 1 <= args.batch <= 8
    from vllm import LLM,SamplingParams
    from transformers import AutoTokenizer
    path=pathlib.Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    cfg=vars(args)|{'input_sha256':hashlib.sha256(pathlib.Path(args.input).read_bytes()).hexdigest(),
                    'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    'enforce_eager':False,'executor_revision':'latest-computed-state-v3c-runtimefix','hint':HINT,'temperature':0.6,'top_p':0.95,'max_tool_rounds':args.max_tool_calls+1,
                    'state_max_tokens_per_call':64,
                    'arms':ARMS, 'ablation_revision':'v3c-three-components-20261005',
                    'protocol':'four arms share prefix until first feedback in any arm; feedback charged to budget'}
    config_path=path.with_suffix('.config.json')
    if config_path.exists():
        assert json.loads(config_path.read_text()) == cfg, 'Refuse resume with different configuration/code'
    else:
        config_path.write_text(json.dumps(cfg,indent=2))
    completed=set()
    if path.exists():
        for s in path.read_text().splitlines():
            r=json.loads(s);completed.add((r['id'],r['sample'],r['action']))
    rows=[json.loads(s) for s in pathlib.Path(args.input).read_text().splitlines()]
    if args.limit: rows=rows[:args.limit]
    tok=AutoTokenizer.from_pretrained(args.model)
    llm=LLM(model=args.model,dtype='bfloat16',max_model_len=args.context,gpu_memory_utilization=0.85,
            max_num_seqs=32,max_num_batched_tokens=8192,enforce_eager=False,seed=args.seed,enable_prefix_caching=True)
    work=[]
    for i,r in enumerate(rows):
        rid=r.get('id',hashlib.sha256(r['problem'].encode()).hexdigest())
        for sample in range(args.samples):
            actions=ARMS
            if all((rid,sample,a) in completed for a in actions):continue
            work.append((i,r,rid,sample))
    for offset in range(0,len(work),args.batch):
        start=time.time();batch=work[offset:offset+args.batch]
        prompts=[tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True) for _,r,_,_ in batch]
        params=[SamplingParams(temperature=.6,top_p=.95,max_tokens=args.prefix,stop=['```python','</think>'],seed=args.seed+i*1009+s*100003) for i,_,_,s in batch]
        prefixes=llm.generate(prompts,params,use_tqdm=False)
        active=[];finished=[]
        for (i,r,rid,s),prompt,out in zip(batch,prompts,prefixes):
            o=out.outputs[0];prefix=o.text
            if isinstance(o.stop_reason,str):prefix+=o.stop_reason
            eligible=o.finish_reason=='length' and '```' not in prefix and '</think>' not in prefix and '\\boxed' not in prefix
            actions=['shared'] if args.mode=='paired' else [args.mode]
            for action in actions:
                if (rid,s,action) in completed:continue
                force=False
                predicted=None
                hint=HINT if force and eligible else ''
                record=dict(id=rid,sample=s,action=action,problem=r['problem'],gt_answer=r['gt_answer'],
                    prefix=prefix,eligible=eligible,intervened=bool(hint),predicted_value=predicted,
                    gen=prefix+hint,prompt=prompt,tokens=len(o.token_ids),hint_tokens=len(tok.encode(hint,add_special_tokens=False)) if hint else 0,
                    calls=0,events=[],last_state={},seed=args.seed+i*1009+s*100003,round=0)
                if o.finish_reason=='stop' and not isinstance(o.stop_reason,str):
                    record['termination']='stop';finished.append(record)
                else:active.append(record)
        for step in range(args.max_tool_calls+1):
            pending=[];params=[];texts=[]
            for r in active:
                remaining=min(args.budget-r['tokens']-r['hint_tokens'],args.context-len(tok.encode(r['prompt']+r['gen'],add_special_tokens=False))-1)
                if remaining<=0:
                    r['termination']='budget';finished.append(r);continue
                pending.append(r);texts.append(r['prompt']+r['gen'])
                params.append(SamplingParams(temperature=.6,top_p=.95,max_tokens=remaining,
                    stop=['`\n'] if step<args.max_tool_calls else [],seed=r['seed']+step+1))
            if not pending:
                active=[]
                break
            outs=llm.generate(texts,params,use_tqdm=False);active=[];jobs=[]
            for r,out in zip(pending,outs):
                o=out.outputs[0];text=o.text
                if isinstance(o.stop_reason,str):text+=o.stop_reason
                r['gen']+=text;r['tokens']+=len(o.token_ids);r['round']=step+1
                if o.stop_reason=='`\n' and step<args.max_tool_calls:
                    code=extract_jupyter_like_program(r['gen'])
                    # A forced/open Python fence can begin in the common prefix.
                    if code.strip():
                        jobs.append((r,code,extract_program(r['gen'],last_only=True)));r['calls']+=1
                    else:active.append(r)
                else:
                    r['termination']=o.finish_reason;finished.append(r)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results=list(pool.map(tool,[{'code':code,'latest_code':latest,'ablation_states':True} for _,code,latest in jobs]))
            for (r,code,latest),result in zip(jobs,results):
                r['events'].append({'code':code,'latest_code':latest,**result})
                output=result['text']
                if step==args.max_tool_calls-1:
                    output+='\n\n[SYSTEM]\nYou have exceeded the allowed number of code executions. You can no longer write or run code. Please continue solving the problem using your reasoning and analytical skills.'
                from cesf.ablation_feedback import feedback_for
                allowance=max(0,min(64,args.budget-r['tokens']-r['hint_tokens']))
                choices=ARMS if r['action']=='shared' else [r['action']]
                prepared={arm:feedback_for(result,r['last_state'],arm,tok,allowance)
                          for arm in choices}
                if r['action']=='shared' and any(x[1] for x in prepared.values()):
                    variants=[copy.deepcopy(r) for _ in ARMS]
                    for v,arm in zip(variants,ARMS):
                        v.update(action=arm,fork_at_call=r['calls'],fork_tokens=r['tokens'])
                else:
                    variants=[r]
                for v in variants:
                    feedback,ids,state=prepared.get(v['action'],('',[],{}))
                    if ids:
                        v['hint_tokens']+=len(ids)
                        v['intervened']=True
                        v['last_state']=state
                        v['events'][-1]['state_feedback']=feedback
                        v['events'][-1]['state_feedback_tokens']=len(ids)
                    v['gen']+='```output\n'+output+feedback+'\n```\n'
                    active.append(v)
        for r in active:r['termination']='round_limit';finished.append(r)
        expanded=[]
        for r in finished:
            if r['action']=='shared':
                for action in ARMS:
                    clone=copy.deepcopy(r);clone['action']=action
                    expanded.append(clone)
            else: expanded.append(r)
        finished=expanded
        with path.open('a') as f:
            for r in finished:
                if (r['id'],r['sample'],r['action']) in completed: continue
                r['correct']=grade(r['gt_answer'],r['gen'])
                r['batch_seconds']=time.time()-start
                del r['prompt']
                f.write(json.dumps(r)+'\n')
                f.flush()
        print(json.dumps({'completed_batch':offset,'records':len(finished),'correct':sum(r['correct'] for r in finished),
              'eligible':sum(r['eligible'] for r in finished),'seconds':round(time.time()-start,1)}),flush=True)
    print('RUN_COMPLETE',flush=True)

if __name__=='__main__':main()
