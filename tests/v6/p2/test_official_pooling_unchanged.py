from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_official_pooling_unchanged():
    source = (ROOT / "vendor" / "tats" / "data_provider" / "data_loader.py").read_text(encoding="utf-8")
    assert "text_embeddings = self.llm_model.get_input_embeddings()(self.input_ids)" in source
    assert "expanded_mask = self.attn_mask.unsqueeze(-1).expand_as(text_embeddings)" in source
    assert "masked_emb = text_embeddings * expanded_mask" in source
    assert "pooled_emb = masked_emb.sum(dim=1) / valid_counts.squeeze(1)" in source
