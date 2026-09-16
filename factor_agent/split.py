"""Train / holdout must stay disjoint by case and by source-document group.

Reports from one institution's factor series recycle wording and parameters, so
splitting by case id alone leaks. The group key falls back from an explicit
`group_id` to `institution::series` to the document id.
"""

from __future__ import annotations

from factor_agent.schemas import FactorCase


def group_key(case: FactorCase) -> str:
    if case.group_id:
        return case.group_id
    document = case.document
    if document is None:
        return case.case_id
    institution = document.institution.strip()
    series = document.series.strip()
    if institution or series:
        return f"{institution or 'unknown'}::{series or 'unknown'}"
    return document.doc_id or case.case_id


def group_map(cases: list[FactorCase]) -> dict[str, str]:
    return {case.case_id: group_key(case) for case in cases}


def split_ids(cases: list[FactorCase]) -> tuple[set[str], set[str]]:
    train = {c.case_id for c in cases if c.split == "train"}
    holdout = {c.case_id for c in cases if c.split == "holdout"}
    return train, holdout


def leak_ids(cases: list[FactorCase]) -> set[str]:
    train, holdout = split_ids(cases)
    return train & holdout


def leak_groups(cases: list[FactorCase]) -> set[str]:
    train = {group_key(c) for c in cases if c.split == "train"}
    holdout = {group_key(c) for c in cases if c.split == "holdout"}
    return train & holdout


def assert_no_leak(cases: list[FactorCase]) -> None:
    leaked = leak_ids(cases)
    if leaked:
        raise ValueError(f"train/holdout leak: {sorted(leaked)}")
    groups = leak_groups(cases)
    if groups:
        raise ValueError(f"train/holdout group leak: {sorted(groups)}")
