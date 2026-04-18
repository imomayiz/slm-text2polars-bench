"""
Prompt routing: classify questions and inject targeted Polars API snippets.

Two signals:
  1. Question text (regex patterns)
  2. Schema (column dtypes, number of tables)

Only injects snippets for APIs where small models commonly fail.
Simple operations (filter, basic agg, sort) are covered by few-shot examples.
"""

from __future__ import annotations

import re
from typing import Any


# ============================================================================
# API snippets — kept minimal to avoid confusing small models
# ============================================================================

API_SNIPPETS = {
    "string": (
        "# Polars string methods use .str namespace with underscores:\n"
        "# pl.col(\"x\").str.starts_with(\"A\")  pl.col(\"x\").str.ends_with(\"z\")\n"
        "# pl.col(\"x\").str.contains(\"foo\")   pl.col(\"x\").str.to_lowercase()\n"
        "# NOT .startswith() or .endswith() — those are Python str methods, not Polars."
    ),
    "groupby_multi": (
        "# Multiple aggs in one group_by — pass a list:\n"
        "# .group_by(\"key\").agg([pl.col(\"x\").mean().alias(\"avg\"), pl.col(\"x\").count().alias(\"n\")])\n"
        "# Filter groups after: .group_by(...).agg(...).filter(pl.col(\"n\") > 2)"
    ),
    "join": (
        "# Join two DataFrames: left.join(right, on=\"key\", how=\"inner\")\n"
        "# how= inner | left | right | full | semi | anti | cross\n"
        "# Anti join for \"not in\" / \"never\": a.join(b, on=\"key\", how=\"anti\")"
    ),
    "window": (
        "# Window (per-group without collapsing rows) — use .over():\n"
        "# .with_columns(pl.col(\"x\").rank(descending=True).over(\"grp\").alias(\"rank\"))\n"
        "# .with_columns((pl.col(\"x\") / pl.col(\"x\").sum()).over(\"grp\").alias(\"pct\"))"
    ),
    "date": (
        "# Date/Datetime columns use .dt namespace:\n"
        "# pl.col(\"d\").dt.year()  .dt.month()  .dt.day()  .dt.weekday()\n"
        "# pl.col(\"d\").dt.strftime(\"%Y-%m\")  — format to string\n"
        "# Filter: pl.col(\"d\") > pl.date(2023, 1, 1)\n"
        "# Duration: (pl.col(\"end\") - pl.col(\"start\")).dt.total_days()"
    ),
    "when_then": (
        "# Conditional column: pl.when(condition).then(value).otherwise(default).alias(\"name\")\n"
        "# Chain: pl.when(c1).then(v1).when(c2).then(v2).otherwise(v3)"
    ),
    "null_nan": (
        "# null is missing for any dtype: .is_null() .fill_null(value) .drop_nulls()\n"
        "# Float NaN is a value, not null: .is_nan() .fill_nan(value)\n"
        "# NOT .fillna() or .isna() — those are pandas."
    ),
    "concat": (
        "# Stack frames: pl.concat([df1, df2], how=\"vertical\")  — rows\n"
        "# Side by side: pl.concat([df1, df2], how=\"horizontal\")  — columns\n"
        "# NOT pandas concat(axis=0/1)."
    ),
}


# ============================================================================
# Pattern matching on question text
# ============================================================================

_ROUTE_PATTERNS: dict[str, list[str]] = {
    "string": [
        r"\bstarts?\s*with\b", r"\bends?\s*with\b", r"\bcontains?\b",
        r"\bsubstring\b", r"\bprefix\b", r"\bsuffix\b",
        r"\bletter\b", r"\bcharacter\b",
        r"\blowercase\b", r"\buppercase\b", r"\blength\b",
        r"\bpattern\b", r"\bregex\b", r"\bmatch\b",
    ],
    "groupby": [
        r"\bper\b", r"\beach\b", r"\bby\b(?!$)",
        r"\bgroup\b", r"\bbreakdown\b", r"\bsummar",
        r"\bfor each\b", r"\bfor every\b",
        r"\bheadcount\b", r"\bdistribution\b",
    ],
    "window": [
        r"\brank\b.*\b(within|per|each|by)\b",
        r"\b(within|per|each)\b.*\brank\b",
        r"\bpercentage\b", r"\bpercent\b", r"\bfraction\b", r"\bshare\b",
        r"\bproportion\b", r"\b%\b",
        r"\brunning\b", r"\bcumulative\b", r"\brolling\b",
        r"\bwindow\b", r"\bover\b",
    ],
    "date": [
        r"\byear\b", r"\bmonth\b", r"\bday\b", r"\bweek\b",
        r"\bdate\b", r"\btime\b", r"\bduration\b",
        r"\bbefore\b", r"\bafter\b", r"\bsince\b", r"\buntil\b",
        r"\bago\b", r"\brecent\b", r"\boldest\b", r"\bnewest\b",
        r"\bquarter\b",
    ],
    "when_then": [
        r"\bif\b.*\bthen\b", r"\bcategorize\b", r"\bclassify\b",
        r"\bbucket\b", r"\bbin\b", r"\blabel\b",
        r"\bcase\b", r"\bcondition\b",
        r"\bhigh|medium|low\b.*\bbased on\b",
        r"\bbased on\b.*\bhigh|medium|low\b",
    ],
    "null_nan": [
        r"\bnull\b", r"\bmissing\b", r"\bnan\b",
        r"\bfillna\b", r"\bdropna\b", r"\bisna\b", r"\bnotna\b",
        r"\bimpute\b", r"\bcoalesce\b",
    ],
    "concat": [
        r"\bconcat\b", r"\bstack\b", r"\bbind\b",
        r"\bappend\b.*\brows?\b", r"\bunion\b",
        r"\bvertically\b", r"\bhorizontally\b",
    ],
    "negation": [
        r"\bnever\b", r"\bnot\b", r"\bno\s+\w+s?\b", r"\bwithout\b",
        r"\bwho have not\b", r"\bwho did not\b", r"\bexclude\b",
    ],
}

# Aggregate words for detecting multi-agg groupby
_AGG_WORDS = {"count", "min", "max", "mean", "average", "sum", "total",
              "headcount", "number", "median", "std"}


# ============================================================================
# Schema-aware detection
# ============================================================================

def _schema_has_dtype(schema: dict[str, dict[str, str]], *prefixes: str) -> bool:
    """Check if any column in any table has a dtype starting with one of the prefixes."""
    for cols in schema.values():
        for dtype in cols.values():
            dt = dtype.lower()
            if any(dt.startswith(p) for p in prefixes):
                return True
    return False


def _is_multi_table(schema: dict[str, dict[str, str]]) -> bool:
    return len(schema) > 1


# ============================================================================
# Main classifier
# ============================================================================

def _text_matches(q: str, category: str) -> bool:
    """Check if question text matches any pattern for a category."""
    return any(re.search(p, q) for p in _ROUTE_PATTERNS.get(category, []))


def classify_question(
    question: str,
    schema: dict[str, dict[str, str]] | None = None,
) -> list[str]:
    """
    Return list of API snippet keys relevant to this question.

    Uses both question text and schema dtypes to decide.
    """
    q = question.lower()
    schema = schema or {}
    hits: list[str] = []

    # --- String ops (text patterns OR string columns mentioned in a string-ish question)
    if _text_matches(q, "string"):
        hits.append("string")
    elif _schema_has_dtype(schema, "utf8", "string", "str", "categorical"):
        # If the schema has string columns and the question mentions specific values
        # with quotes or name-like filters, hint string API
        if re.search(r"['\"]|named?\b|called\b", q):
            hits.append("string")

    # --- Multi-agg groupby
    if _text_matches(q, "groupby"):
        agg_count = sum(1 for w in _AGG_WORDS if w in q)
        if agg_count >= 2:
            hits.append("groupby_multi")

    # --- Joins (multi-table schema is the strongest signal)
    if _is_multi_table(schema):
        hits.append("join")
    elif _text_matches(q, "negation") and _is_multi_table(schema):
        # Anti-join: negation + multiple tables
        hits.append("join")

    # --- Window functions
    if _text_matches(q, "window"):
        hits.append("window")

    # --- Date/time (text patterns OR date columns in schema)
    if _text_matches(q, "date"):
        hits.append("date")
    elif _schema_has_dtype(schema, "date", "datetime", "time", "duration"):
        hits.append("date")

    # --- Conditional / when-then
    if _text_matches(q, "when_then"):
        hits.append("when_then")

    # --- Null/NaN handling
    if _text_matches(q, "null_nan"):
        hits.append("null_nan")

    # --- Concat/stack
    if _text_matches(q, "concat"):
        hits.append("concat")

    return hits
