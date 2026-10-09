import importlib.util,json,pathlib,sys,tempfile,types
root=pathlib.Path(__file__).resolve().parents[1];calls=[]
sys.path.insert(0,str(root))
sys.modules["infer"]=types.ModuleType("infer")
parser=types.ModuleType("infer.parser")
parser.extract_jupyter_like_program=lambda text: text
parser.extract_program=lambda text,**kwargs: text
sys.modules["infer.parser"]=parser
class Tok:
    def apply_chat_template(self,messages,**kw):return messages[0]['content']
    def encode(self,text,**kw):return list(text)
class AT:
    @staticmethod
    def from_pretrained(path):return Tok()
def out(text,finish='stop',stop=None):
    return types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text,
        finish_reason=finish,stop_reason=stop,token_ids=[0]*len(text))])
class LLM:
    def __init__(self,**kw):pass
    def generate(self,texts,params,**kw):
        calls.append(len(texts))
        if len(calls)==1:return [out('Thought.','length') for t in texts]
        if len(calls)==2:
            assert len(texts)==2
            return [out('\n```python\nresult=21\nprint(result)\n``','stop','`\n'),out(' done')]
        assert len(texts)==2
        assert sum('[Python state update' in t for t in texts)==1
        return [out(' final') for t in texts]
class SP:
    def __init__(self,**kw):pass
sys.modules['vllm']=types.SimpleNamespace(LLM=LLM,SamplingParams=SP)
sys.modules['transformers']=types.SimpleNamespace(AutoTokenizer=AT)
spec=importlib.util.spec_from_file_location('runner',root/'reproduction/run_paired.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.grade=lambda a,b:False
m.tool=lambda code:{'text':'21','ok':True,'state':{'result':'21'}}
with tempfile.TemporaryDirectory() as d:
    source=pathlib.Path(d)/'input.jsonl';target=pathlib.Path(d)/'out.jsonl'
    source.write_text(''.join(json.dumps({'id':str(i),'problem':'p','prompt':'P'+str(i),
        'gt_answer':'0'})+'\n' for i in range(2)))
    sys.argv=['runner','--input',str(source),'--output',str(target),'--samples','1','--model','mock-model','--seed','42']
    m.main()
    rows=[json.loads(line) for line in target.read_text().splitlines()]
    g={(r['id'],r['action']):r for r in rows}
    assert len(g)==len(rows)==4 and calls==[2,2,2]
    a,b=g[('0','natural')],g[('0','state_sync')]
    assert not a['intervened'] and b['intervened']
    assert a['fork_at_call']==b['fork_at_call']==1
    assert a['fork_tokens']==b['fork_tokens']
    assert b['hint_tokens']>0 and a['hint_tokens']==0
    assert a['events'][0]['code']==b['events'][0]['code']
    a,b=g[('1','natural')],g[('1','state_sync')]
    assert a['gen']==b['gen'] and not b['intervened']
print('STATE_RUNNER_PASS: shared trajectory, actual fork, exact no-op, feedback cost')
