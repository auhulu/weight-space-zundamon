"""Checks that prevent leakage and prompt contamination."""
import unittest
from unittest.mock import patch
import experiment

class FakeBatch(dict):
    def to(self,device): return self
class FakeTokenizer:
    def apply_chat_template(self,messages,**kwargs):
        self.messages=messages; self.kwargs=kwargs
        return FakeBatch()

class ContractTests(unittest.TestCase):
    def test_plain_has_no_style_instruction(self):
        tok=FakeTokenizer()
        experiment.inputs(tok,'質問',False)
        self.assertEqual(tok.messages,[{'role':'system','content':experiment.CFG['system']},{'role':'user','content':'質問'}])
        self.assertFalse(tok.kwargs['enable_thinking'])
    def test_style_only_in_prompt_baseline(self):
        tok=FakeTokenizer()
        experiment.inputs(tok,'質問',True)
        self.assertIn(experiment.CFG['style'],tok.messages[0]['content'])
    def test_stages_are_disjoint(self):
        self.assertEqual(experiment.CFG['final_seeds'],[101,202,303])
        self.assertNotIn(experiment.CFG['selection_seed'],experiment.CFG['final_seeds'])

if __name__=='__main__':unittest.main()
