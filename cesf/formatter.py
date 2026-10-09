"""Small integration API around the unchanged experimental formatter."""
import ast
from dataclasses import dataclass
from typing import Dict, List, Mapping, Any
from .execution_state import snapshot, render_delta, bounded_feedback


@dataclass(frozen=True)
class Feedback:
    output: str
    text: str
    token_ids: List[int]
    candidates: Dict[str, str]


class CESF:
    """Create one instance per reasoning trajectory; this class executes no code.

    tokenizer must provide encode(text, add_special_tokens=False).
    As in the frozen runner, a successful emission saves the full candidate
    snapshot, including candidates not emitted because of line/token caps.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.previous = {}

    def reset(self):
        """Start a new trajectory without carrying state across questions."""
        self.previous = {}

    def format(self, latest_code: str, namespace: Mapping[str, Any],
               stdout: str = '', remaining_tokens: int = 64,
               execution_ok: bool = True) -> Feedback:
        if not execution_ok:
            return Feedback(stdout, '', [], {})
        try:
            current = snapshot(ast.parse(latest_code), namespace)
        except (SyntaxError, ValueError, TypeError):
            return Feedback(stdout, '', [], {})
        text, ids = bounded_feedback(
            render_delta(current, self.previous), self.tokenizer,
            max(0, min(64, int(remaining_tokens))))
        if text:
            self.previous = current.copy()
        return Feedback(stdout + text, text, ids, current)
