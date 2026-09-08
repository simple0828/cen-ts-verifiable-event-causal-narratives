from __future__ import annotations

import re


def validate_narrative(text: str, event: dict, verification: dict) -> list[str]:
    errors: list[str] = []
    for field in ["event_type", "report_time", "evidence_span"]:
        value = str(event.get(field, ""))
        if value and value not in text:
            errors.append(f"missing_event_field:{field}")
    for field in ["corrected_direction", "corrected_lag_min", "corrected_lag_max"]:
        value = str(verification.get(field, ""))
        if value and value not in text:
            errors.append(f"missing_verifier_field:{field}")
    allowed_numbers = {str(event.get("expected_duration", "")), str(verification.get("corrected_lag_min", "")), str(verification.get("corrected_lag_max", ""))}
    for num in re.findall(r"\b\d+(?:\.\d+)?\b", text):
        if num not in allowed_numbers and not re.match(r"^\d{4}$", num):
            continue
    return errors
