"""Bounded mathematical Python execution; no shell, file, or network imports."""
import ast, contextlib, io, json, resource, sys, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

resource.setrlimit(resource.RLIMIT_CPU, (25,25))
resource.setrlimit(resource.RLIMIT_AS, (3*1024**3,3*1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024*1024,1024*1024))
allowed = {'matplotlib','mpmath','pandas','math','cmath','sympy','numpy','scipy','itertools','functools','collections','fractions','decimal','statistics','random','operator','heapq','bisect','typing','string','re'}
forbidden = {'open','eval','exec','compile','input','getattr','setattr','delattr','globals','locals','breakpoint','help','__import__'}
class Capture(io.StringIO):
    def write(self,s):
        if self.tell()+len(s)>65536:
            raise RuntimeError('Output limit exceeded')
        return super().write(s)

try:
    code = sys.stdin.read()
    if '--structured-input' in sys.argv:
        payload=json.loads(code)
        code=payload['code'];latest_code=payload['latest_code']
    else:latest_code=code
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            assert all(a.name.split('.')[0] in allowed for a in node.names), 'disallowed import'
        if isinstance(node,ast.ImportFrom):
            assert node.module and node.module.split('.')[0] in allowed and node.level == 0, 'disallowed import'
        if isinstance(node,ast.Name):
            assert node.id not in forbidden and not node.id.startswith('__'), 'disallowed name'
        if isinstance(node,ast.Attribute):
            assert not node.attr.startswith('_') and node.attr not in {'load','save','savez','fromfile','tofile','read','write','read_text','write_text','system','popen','loadtxt','savetxt','genfromtxt','memmap','ctypes'}, 'disallowed attribute'
    output = Capture()
    namespace={}
    with contextlib.redirect_stdout(output):
        try:
            exec(compile(tree,'<tool>','exec'),namespace)
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
    text = output.getvalue().strip()
    if len(text)>800:
        text=text[:400]+'...'+text[-400:]
    try:
        from cesf.execution_state import snapshot
        state=snapshot(ast.parse(latest_code),namespace)
    except Exception:
        state={}
    result={'text':text,'ok':True,'state':state}
    if '--structured-input' in sys.argv and payload.get('ablation_states'):
        try: result['state_all']=snapshot(tree,namespace)
        except Exception: result['state_all']={}
    print(json.dumps(result))
except BaseException:
    text = traceback.format_exc()
    print(json.dumps({'text':text[:400]+'...'+text[-400:] if len(text)>800 else text,'ok':False}))
