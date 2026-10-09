"""Single-component removals; keep candidate/line/token caps unchanged."""
from .execution_state import render_delta, bounded_feedback

ARMS = ['full', 'no_latest', 'no_delta', 'values_only']

def feedback_for(result, previous, arm, tokenizer, allowance):
    assert arm in ARMS
    if not result.get('ok'):
        return '', [], {}
    current = result.get('state_all' if arm == 'no_latest' else 'state', {})
    if arm == 'values_only':
        delta = [v for k, v in current.items() if previous.get(k) != v]
        text = '\n[Python state update]\n' + '\n'.join(delta[:3]) if delta else ''
    else:
        text = render_delta(current, {} if arm == 'no_delta' else previous)
    text, ids = bounded_feedback(text, tokenizer, allowance)
    return text, ids, current
