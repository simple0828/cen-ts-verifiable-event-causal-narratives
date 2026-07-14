from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import TextVariantRecord
from .text_modes import TextMode, parse_text_mode


CONSTANT_TEXT = "No information available."
UNIMPLEMENTED_MODES = {
    TextMode.EVENT,
    TextMode.CAUSAL_EVENT,
    TextMode.VERIFIED_EVENT,
    TextMode.APO_EVENT,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_hash(df: pd.DataFrame) -> str:
    return sha256_bytes(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def split_ranges(n_rows: int) -> dict[str, tuple[int, int]]:
    num_train = int(n_rows * 0.7)
    num_test = int(n_rows * 0.2)
    num_val = n_rows - num_train - num_test
    return {
        "train": (0, num_train),
        "val": (num_train, num_train + num_val),
        "test": (num_train + num_val, n_rows),
    }


class RawTextProcessor:
    mode = TextMode.RAW

    def transform(self, source_text: str, timestamp: str, context: dict[str, Any]) -> TextVariantRecord:
        return TextVariantRecord(timestamp=timestamp, source_text=source_text, text=source_text, mode=self.mode.value, text_column=context["text_column"])


class ConstantTextProcessor:
    mode = TextMode.CONSTANT

    def transform(self, source_text: str, timestamp: str, context: dict[str, Any]) -> TextVariantRecord:
        return TextVariantRecord(timestamp=timestamp, source_text=source_text, text=CONSTANT_TEXT, mode=self.mode.value, text_column=context["text_column"])


class ShuffledTextProcessor:
    mode = TextMode.SHUFFLED

    def transform(self, source_text: str, timestamp: str, context: dict[str, Any]) -> TextVariantRecord:
        shuffled_text = context.get("shuffled_text")
        if shuffled_text is None:
            raise ValueError("ShuffledTextProcessor requires context['shuffled_text'].")
        return TextVariantRecord(timestamp=timestamp, source_text=source_text, text=shuffled_text, mode=self.mode.value, text_column=context["text_column"])


def build_text_variant(
    source_csv: Path,
    output_csv: Path,
    mode: str,
    text_column: str,
    seed: int = 2025,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    text_mode = parse_text_mode(mode)
    if text_mode in UNIMPLEMENTED_MODES:
        raise NotImplementedError(f"text mode {text_mode.value!r} is reserved for a later stage.")

    df = pd.read_csv(source_csv)
    if "fact" not in df.columns:
        raise ValueError("source CSV must contain a fact column.")
    before = df.copy(deep=True)
    source_hash = sha256_file(source_csv)
    non_text_columns = [col for col in df.columns if col != text_column]
    non_text_identity_before = frame_hash(before[[col for col in before.columns if col != text_column]])
    permutation_mapping: list[dict[str, int | str]] = []

    if text_column not in df.columns:
        df[text_column] = ""

    source_text = before["fact"].fillna(CONSTANT_TEXT)

    if text_mode == TextMode.RAW:
        df[text_column] = source_text
    elif text_mode == TextMode.CONSTANT:
        df[text_column] = CONSTANT_TEXT
    elif text_mode == TextMode.SHUFFLED:
        for split_name, (start, end) in split_ranges(len(df)).items():
            split_index = list(range(start, end))
            shuffled_index = pd.Series(split_index).sample(frac=1.0, random_state=seed).tolist()
            df.loc[split_index, text_column] = source_text.loc[shuffled_index].to_numpy()
            permutation_mapping.extend(
                {"split": split_name, "target_row": int(dst), "source_row": int(src)}
                for dst, src in zip(split_index, shuffled_index)
            )
    else:
        raise NotImplementedError(f"text mode {text_mode.value!r} is reserved for a later stage.")

    changed_columns = [col for col in df.columns if col not in before.columns or not df[col].equals(before[col])]
    allowed_changed = {text_column}
    if set(changed_columns) - allowed_changed:
        raise AssertionError(f"Only {text_column!r} may change, but changed columns were {changed_columns}.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    non_text_identity_after = frame_hash(df[[col for col in df.columns if col != text_column]])
    manifest = {
        "mode": text_mode.value,
        "source_csv": str(source_csv),
        "output_csv": str(output_csv),
        "text_column": text_column,
        "seed": seed,
        "row_count": int(len(df)),
        "source_sha256": source_hash,
        "output_sha256": sha256_file(output_csv),
        "source_non_text_identity_sha256": non_text_identity_before,
        "output_non_text_identity_sha256": non_text_identity_after,
        "non_text_columns_identical": non_text_identity_before == non_text_identity_after,
        "changed_columns": changed_columns,
        "split_ranges": {key: [start, end] for key, (start, end) in split_ranges(len(df)).items()},
        "permutation_mapping": permutation_mapping,
        "example_record": asdict(
            RawTextProcessor().transform(str(before["fact"].iloc[0]), str(before["date"].iloc[0]), {"text_column": text_column})
        ),
    }
    if manifest_path:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return manifest
