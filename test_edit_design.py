import json
import unittest
import torch
from edit_design import apply_edit, design_candidates


class EditAlgebraTest(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList()
        self.rows = []
        for i in range(8):
            block = torch.nn.Module()
            block.mlp = torch.nn.Module()
            block.mlp.down_proj = torch.nn.Linear(3, 2, bias=False)
            self.model.layers.append(block)
            self.rows.append(dict(layer=i, module_name=f"layers.{i}.mlp.down_proj",
                                  h_mean=torch.tensor([1., 2., 3.]),
                                  h_second_moment=torch.tensor([2., 8., 12.]),
                                  d_mean=torch.tensor([.01, .02]) * (i + 1),
                                  d_rms=.3, residual_rms=10.))
        self.stats = {"layers": self.rows, "train_count": 24}

    def test_exact_mean_action_and_scope(self):
        for candidate in design_candidates(self.stats):
            originals = {n: p.clone() for n, p in self.model.named_parameters()}
            metadata = apply_edit(self.model, self.stats, candidate)
            json.dumps(candidate)
            json.dumps(metadata)
            names = {e["module_name"] + ".weight" for e in candidate["edits"]}
            for name, parameter in self.model.named_parameters():
                if name not in names:
                    self.assertTrue(torch.equal(originals[name], parameter))
            for detail in metadata["edits"]:
                row = self.rows[detail["layer"]]
                name = detail["module_name"] + ".weight"
                change = dict(self.model.named_parameters())[name] - originals[name]
                torch.testing.assert_close(change @ row["h_mean"], row["d_mean"] * detail["scale"], atol=2e-7, rtol=1e-5)
            self.assertGreater(metadata["changed_elements"], 0)
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    parameter.copy_(originals[name])

    def test_actual_half_precision_count(self):
        self.model.half()
        before = {n: p.clone() for n, p in self.model.named_parameters()}
        metadata = apply_edit(self.model, self.stats, design_candidates(self.stats)[0])
        count = sum(int((p != before[n]).sum()) for n, p in self.model.named_parameters())
        self.assertEqual(metadata["changed_elements"], count)


if __name__ == "__main__":
    unittest.main()
