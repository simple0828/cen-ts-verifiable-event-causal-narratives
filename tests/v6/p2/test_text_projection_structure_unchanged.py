from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_text_projection_structure_unchanged():
    source = (ROOT / "vendor" / "tats" / "exp" / "exp_long_term_forecasting.py").read_text(encoding="utf-8")
    block = source[source.index("self.mlp = nn.Sequential(") : source.index("# print number of parameters of self.model")]
    assert "nn.Linear(mlp_sizes[0], mlp_sizes[1])" in block
    assert "nn.ReLU()" in block
    assert "nn.Linear(mlp_sizes[1], mlp_sizes[2])" in block
    assert "nn.Dropout(0.3)" in block
