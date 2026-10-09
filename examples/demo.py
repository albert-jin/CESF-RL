"""No model, network, or external packages are needed for this demo."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cesf import CESF


class CharacterTokenizer:
    """Character-count stand-in for a runnable demo, not a model tokenizer."""
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def main():
    formatter = CESF(CharacterTokenizer())
    # The hosting interpreter already executed this trusted example.
    code = 'x = 17\ny = 4\nresult = x + y\nprint(result)'
    namespace = {'x': 17, 'y': 4, 'result': 21}
    first = formatter.format(code, namespace, stdout='21')
    print(first.output)
    assert first.text == '\n[Python state update]\nresult = 21'
    assert len(first.token_ids) <= 64
    second = formatter.format(code, namespace, stdout='21')
    assert second.output == '21' and not second.text
    print('\nRepeated state: no additional feedback.')


if __name__ == '__main__':
    main()
