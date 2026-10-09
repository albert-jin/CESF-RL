import ast
from fractions import Fraction
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cesf import CESF, snapshot, describe, render_delta, bounded_feedback
from cesf.ablation_feedback import feedback_for


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode('utf-8'))


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer()

    def test_latest_only(self):
        got = snapshot(ast.parse('answer = x + y'),
                       {'old': 99, 'x': 17, 'y': 4, 'answer': 21})
        self.assertEqual(got, {'answer': '21'})

    def test_literal_and_container_assignments_are_excluded(self):
        code = 'a=1\nb=a\nc=[1,2]\nd=(1,2)\ne={1:2}\nf={1,2}'
        self.assertEqual(snapshot(ast.parse(code), dict(a=1,b=1,c=[1,2],d=(1,2),e={1:2},f={1,2})), {})

    def test_augmented_and_annotated_assignments(self):
        self.assertEqual(snapshot(ast.parse('x += 1\ny: int = sum([2,3])'), {'x':3,'y':5}), {'y':'5','x':'3'})

    def test_print_mentions_and_reverse_order(self):
        got = snapshot(ast.parse('a=sum([1])\nb=sum([2])\nc=sum([3])\nprint(a)'), {'a':1,'b':2,'c':3})
        self.assertEqual(list(got), ['a','c','b'])

    def test_six_supported_bindings_after_filtering(self):
        code='\n'.join('x%d=sum([1])'%i for i in range(8))+'\n_private=sum([1])\nmissing=sum([1])'
        got=snapshot(ast.parse(code), dict({'x%d'%i:i for i in range(8)}, _private=1))
        self.assertEqual(list(got), ['x7','x6','x5','x4','x3','x2'])

    def test_representation_matches_frozen_code(self):
        self.assertEqual(describe(Fraction(27,26)), '27/26')
        self.assertIsNone(describe('short string'))
        self.assertIsNone(describe(10**150))
        self.assertIn('unique=1', describe([1]*12))

    def test_delta_and_three_binding_cap(self):
        got=render_delta({'a':'1','b':'2','c':'3','d':'4','e':'5'},{'a':'1'})
        self.assertEqual(got.splitlines()[2:], ['b = 2','c = 3','d = 4'])

    def test_whole_line_packing_skips_long_line(self):
        text='\n[Python state update]\nlong = '+'1'*80+'\nx = 1'
        got,ids=bounded_feedback(text,self.tok,40)
        self.assertNotIn('long =',got)
        self.assertIn('x = 1',got)
        self.assertLessEqual(len(ids),40)

    def test_header_alone_is_not_sent(self):
        self.assertEqual(bounded_feedback('\n[Python state update]\nx = 1',self.tok,2), ('',[]))

    def test_stdout_unchanged_and_repeated_state_suppressed(self):
        f=CESF(self.tok);code='x=sum([1,2])';stdout='  3\n'
        first=f.format(code,{'x':3},stdout)
        self.assertTrue(first.output.startswith(stdout))
        self.assertEqual(f.format(code,{'x':3},stdout).output,stdout)

    def test_failed_execution_and_parse_keep_snapshot(self):
        f=CESF(self.tok);f.previous={'a':'1'}
        self.assertEqual(f.format('bad code !',{},'error').output,'error')
        self.assertFalse(f.format('x=sum([1])',{'x':1},execution_ok=False).text)
        self.assertEqual(f.previous,{'a':'1'})

    def test_no_emission_does_not_update_snapshot(self):
        f=CESF(self.tok)
        self.assertFalse(f.format('x=sum([1])',{'x':1},remaining_tokens=0).text)
        self.assertEqual(f.previous,{})

    def test_full_candidate_snapshot_is_saved_on_emission(self):
        f=CESF(self.tok)
        code='\n'.join('x%d=sum([1])'%i for i in range(5))
        got=f.format(code,{'x%d'%i:i for i in range(5)})
        self.assertEqual(len(got.text.splitlines()[2:]),3)
        self.assertEqual(len(f.previous),5)
        self.assertFalse(f.format(code,{'x%d'%i:i for i in range(5)}).text)
        f.reset();self.assertEqual(f.previous,{})

    def test_ablation_isolation(self):
        result={'ok':True,'state':{'answer':'42','half':'21'},'state_all':{'old':'99','answer':'42','half':'21'}}
        previous={'answer':'42'}
        full=feedback_for(result,previous,'full',self.tok,64)[0]
        self.assertNotIn('answer =',full)
        self.assertIn('answer =',feedback_for(result,previous,'no_delta',self.tok,64)[0])
        self.assertIn('old =',feedback_for(result,previous,'no_latest',self.tok,64)[0])
        self.assertNotIn('half',feedback_for(result,previous,'values_only',self.tok,64)[0])


if __name__ == '__main__':
    unittest.main()
