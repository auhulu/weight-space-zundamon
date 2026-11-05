import json
import unittest
from pathlib import Path
from scoring import score, remove_quotes, validate_side


class ScoringTests(unittest.TestCase):
    def test_endings(self):
        s = score('晴れなのだ。散歩がおすすめなのだ！水も持っていきます。')
        self.assertEqual(s['sentences'], 3)
        self.assertEqual(s['target_endings'], 2)
    def test_quotes(self):
        self.assertFalse(score('「なのだ」は語尾です。')['target_mention'])
        self.assertEqual(remove_quotes('前「外『中』外」後'), '前後')
        self.assertEqual(score('「言葉」を説明するのだ。')['target_endings'], 1)
    def test_false_ending(self):
        self.assertEqual(score('なのだと思います。')['target_endings'], 0)
        self.assertTrue(score('なのだと思います。')['target_mention'])
    def test_degenerate(self):
        self.assertTrue(score('  ')['empty'])
        self.assertTrue(score('。！？...')['empty'])
        self.assertTrue(score('なのだ。のだ！')['style_only'])
        self.assertTrue(score('水です。水です。水です。')['repetition'])
        self.assertFalse(score('水を飲むのだ。')['style_only'])
    def test_fenced(self):
        s = score('```text\nなのだ。\n```\n説明です。')
        self.assertEqual(s['sentences'], 1)
        self.assertFalse(s['target_mention'])
    def test_side(self):
        self.assertTrue(validate_side('{"x": 1}', {'kind':'json_equal','value':{'x':1}}))
        self.assertFalse(validate_side('```json\n{"x":1}\n```', {'kind':'json_equal','value':{'x':1}}))
        self.assertTrue(validate_side('便利です。使えます。', {'kind':'polite'}))
        self.assertFalse(validate_side('便利なのだ。', {'kind':'polite'}))
    def test_dataset(self):
        root = Path(__file__).parent / 'data'
        seen = set()
        for split, size in [('train',24),('validation',24),('test',48),('side',16)]:
            rows = json.loads((root / f'{split}.json').read_text(encoding='utf-8'))
            self.assertEqual(len(rows),size)
            for row in rows:
                self.assertNotIn(row['prompt'],seen)
                seen.add(row['prompt'])
            if split != 'side':
                self.assertEqual({sum(r['category']==c for r in rows) for c in ['daily','knowledge','explanation','advice']},{size//4})


if __name__ == '__main__':
    unittest.main()
