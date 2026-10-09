"""Compact newly computed bindings from the latest code block only."""
import ast

def describe(value,depth=0):
    t=type(value)
    if t in (int,float,complex,bool):
        text=repr(value)
        return text if len(text)<=80 else None
    if t.__module__ in ('fractions','decimal'):
        text=str(value)
        return text if len(text)<=80 else None
    if t.__module__.startswith('sympy.'):
        if value.free_symbols:return None
        text=str(value)
        return text if len(text)<=80 else None
    if t.__module__.startswith('numpy') and hasattr(value,'shape'):
        if value.shape==():return describe(value.item(),depth+1)
        if value.size<=8:return describe(value.tolist(),depth+1)
        return None
    if t in (list,tuple) and depth<2:
        desc=[describe(x,depth+1) for x in value[:8]]
        if len(value)<=8 and all(x is not None for x in desc):
            text='['+', '.join(desc)+']'
            if len(text)<=80:return text
        text=f'{t.__name__}(length={len(value)}'
        if len(value)<=2048:
            try:text+=f', unique={len(set(value))}'
            except (TypeError,ValueError):pass
        return text+')'
    return None

def snapshot(tree,namespace):
    computed=[];printed=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
            if not isinstance(node,ast.AugAssign) and isinstance(node.value,(ast.Constant,ast.Name,ast.List,ast.Tuple,ast.Dict,ast.Set)):
                continue
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            computed.extend(n.id for t in targets for n in ast.walk(t) if isinstance(n,ast.Name))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='print':
            printed.extend(n.id for arg in node.args for n in ast.walk(arg) if isinstance(n,ast.Name))
    names=list(dict.fromkeys([n for n in printed if n in computed]+list(reversed(computed))))
    values={}
    for name in names:
        if name.startswith('_') or name not in namespace:continue
        try:text=describe(namespace[name])
        except Exception:continue
        if text is not None:values[name]=text
        if len(values)>=6:break
    return values

def render_delta(current,previous):
    delta={k:v for k,v in current.items() if previous.get(k)!=v}
    if not delta:return ''
    return '\n[Python state update]\n'+'\n'.join(f'{k} = {v}' for k,v in list(delta.items())[:3])

def bounded_feedback(text,tokenizer,allowance):
    if not text or allowance<=0:return '',[]
    lines=text.splitlines();selected=lines[:2];included=0
    for line in lines[2:]:
        trial='\n'.join(selected+[line])
        if len(tokenizer.encode(trial,add_special_tokens=False))<=allowance:
            selected.append(line);included+=1
    if not included:return '',[]
    result='\n'.join(selected)
    return result,tokenizer.encode(result,add_special_tokens=False)
