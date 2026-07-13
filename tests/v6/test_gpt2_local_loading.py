import torch

from cen_tats.runtime.preflight import check_gpt2_files, encode_texts_masked_average, load_pretrained_gpt2_strict


def test_gpt2_path_exists_and_hidden_size_is_768() -> None:
    manifest = check_gpt2_files("D:/models/gpt2")
    assert manifest["complete"] is True
    _, _, meta = load_pretrained_gpt2_strict("D:/models/gpt2", local_files_only=True)
    assert meta["hidden_size"] == 768
    assert meta["random_init"] is False


def test_local_files_only_embeddings_are_nonconstant() -> None:
    tokenizer, model, _ = load_pretrained_gpt2_strict("D:/models/gpt2", local_files_only=True)
    embeddings = encode_texts_masked_average(
        tokenizer,
        model,
        [
            "A major oil producer announced a production cut.",
            "A severe weather event disrupted regional energy supply.",
            "No material event was reported.",
        ],
    )
    assert float(embeddings.var(unbiased=False).item()) > 0
    assert not torch.allclose(embeddings[0], embeddings[1])
