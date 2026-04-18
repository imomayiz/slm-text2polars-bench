"""
Shared prompt routing: targeted API snippets + classify_question(...).

Used by inference.py and generator.py so the CUDA bench path does not need generator.py.

Documentation sources (Polars vs pandas)
----------------------------------------
- Python API index (all public ``polars.*`` objects):
  https://docs.pola.rs/api/python/stable/reference/index.html
- Lazy API (``LazyFrame``, ``collect()``, query plan, predicate/projection pushdown):
  https://docs.pola.rs/user-guide/concepts/lazy-api/
- Expressions and contexts (``Expr``, ``select`` / ``with_columns`` / ``filter`` / ``group_by``):
  https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/
- Data types and Arrow columnar layout:
  https://docs.pola.rs/user-guide/concepts/data-types-and-structures/
- Missing data (``null`` vs float ``NaN``, ``is_null`` / ``fill_null`` vs ``is_nan`` / ``fill_nan``):
  https://docs.pola.rs/user-guide/expressions/missing-data/
- Migration from pandas (no index, no ``.loc``/``.iloc``, expression DSL):
  https://docs.pola.rs/user-guide/migration/pandas/
- Joins (``how`` including ``semi`` / ``anti``, ``join_asof``, ``join_where``):
  https://docs.pola.rs/user-guide/transformations/joins/
- Concatenation (``pl.concat``, ``how=vertical|horizontal|diagonal`` — not pandas ``axis=``):
  https://docs.pola.rs/user-guide/transformations/concatenation/

This module keeps *conditional* ``API_SNIPPETS`` small (token budget); static reference
text lives in ``POLARS_API_REFERENCE_NOTES`` for maintainers and optional reuse.
"""

from __future__ import annotations

import re
from typing import Any

# Dense reference aligned with the Polars docs above (pandas contrasts in-line).
# Not injected into every prompt by default — see API_SNIPPETS + classify_question.
POLARS_API_REFERENCE_NOTES = """
Polars vs pandas (API reference themes):
- No row index: use integer position; there is no MultiIndex / .loc / .iloc; avoid index-shaped logic.
- Expressions: build lazy Expr (e.g. pl.col, .when/.then) and run them inside contexts: select, with_columns, filter, group_by/agg — not ad-hoc .apply for everything.
- LazyFrame vs DataFrame: lazy builds a query plan; collect() executes; explain() shows optimization. (Bench code here uses eager DataFrame only.)
- String API: Utf8/String columns use the .str namespace; methods use underscores (.str.starts_with), not pandas-style .str.startswith on Series.
- Missing values: null is the missing sentinel for all dtypes; float NaN is a valid float value, not null — use is_null/fill_null vs is_nan/fill_nan; aggregates skip nulls by default.
- Joins: join(..., how= inner|left|right|full|semi|anti|cross); semi/anti for filter-by-membership; join_asof / join_asof_by for as-of; join_where for non-equi.
- Concat: pl.concat(..., how= vertical|horizontal|diagonal), not pandas concat(axis=0/1) naming.
- Group by: group_by(keys).agg(...); multiple aggregations as a list of Expr; filter groups with .filter on the result.
- Windows: Expr.over(...) for per-group vector values without collapsing rows (transform-style).
""".strip()


# Only inject snippets for tricky APIs where small models commonly fail.
# Simple categories (filter, agg, sort) are already well-covered by few-shot examples.
API_SNIPPETS = {
    "string": (
        "# Polars uses underscores: .str.starts_with() .str.ends_with() .str.contains()\n"
        "# NOT .startswith() or .ends_with()"
    ),
    "groupby_multi": (
        "# Multiple aggs: .agg([pl.col(\"x\").count().alias(\"n\"), pl.col(\"x\").min().alias(\"min_x\")])\n"
        "# Filter after groupby: .group_by(...).agg(...).filter(pl.col(\"n\") > 2)"
    ),
    "join_anti": (
        "# \"never ordered\" / \"no match\" = anti join: a.join(b, on=\"key\", how=\"anti\")"
    ),
    "window": (
        "# Rank within group: .with_columns(pl.col(\"x\").rank(descending=True).over(\"grp\").alias(\"rank\"))\n"
        "# Pct of total: .with_columns((pl.col(\"x\") / pl.col(\"x\").sum() * 100).alias(\"pct\"))"
    ),
    "null_nan": (
        "# Missing: use is_null() / fill_null() — null is missing for any dtype.\n"
        "# Float NaN is a value, not null: is_nan() / fill_nan() (see missing-data user guide)."
    ),
    "concat_stack": (
        "# Stack/bind frames: pl.concat(frames, how=\"vertical\") for rows; how=\"horizontal\" for columns.\n"
        "# Diagonal concat for union-by-name; not pandas concat(axis=0|1)."
    ),
}

# Patterns that trigger each snippet category — general-purpose, not bench-specific
_ROUTE_PATTERNS: dict[str, list[str]] = {
    "filter": [
        r"\bwhere\b", r"\bfilter\b", r"\bonly\b", r"\bexclude\b",
        r"\bwho\s+(have|has|are|is|did|do|were|was)\b",
        r"\bwhich\b", r"\bif\b",
        r"\babove\b", r"\bbelow\b", r"\bgreater\b", r"\bless\b",
        r"\bbetween\b", r"\bmore than\b", r"\bfewer than\b", r"\bat least\b",
        r"\bnot\b", r"\bnever\b", r"\bwithout\b",
        r"[><=!]+\s*\d",  # numeric comparisons in the question
    ],
    "string": [
        r"\bstarts?\s*with\b", r"\bends?\s*with\b", r"\bcontains?\b",
        r"\bsubstring\b", r"\bprefix\b", r"\bsuffix\b",
        r"\bletter\b", r"\bcharacter\b", r"\btext\b",
        r"\blowercase\b", r"\buppercase\b", r"\blength\b",
        r"\bpattern\b", r"\bregex\b", r"\bmatch\b",
    ],
    "groupby": [
        r"\bper\b", r"\beach\b", r"\bby\b(?!$)",
        r"\bgroup\b", r"\bbreakdown\b", r"\bsummar",
        r"\bfor each\b", r"\bfor every\b",
        r"\bheadcount\b", r"\bdistribution\b",
    ],
    "join": [
        r"\bjoin\b", r"\bmerge\b", r"\bcombine\b", r"\blookup\b",
        r"\bnever\s+(placed|ordered|bought|made)\b",
        r"\bwho have not\b", r"\bwho did not\b", r"\bno\s+orders?\b",
    ],
    "sort": [
        r"\bsort\b", r"\border\b", r"\brank\b",
        r"\btop\s+\d", r"\bbottom\s+\d", r"\bfirst\s+\d", r"\blast\s+\d",
        r"\blargest\b", r"\bsmallest\b", r"\bhighest\b", r"\blowest\b",
        r"\bdescending\b", r"\bascending\b",
        r"\bmost\b", r"\bfewest\b", r"\bbiggest\b",
    ],
    "window": [
        r"\brank\b.*\b(within|per|each|by)\b",
        r"\b(within|per|each)\b.*\brank\b",
        r"\bpercentage\b", r"\bpercent\b", r"\bfraction\b", r"\bshare\b",
        r"\bproportion\b", r"\b%\b",
        r"\brunning\b", r"\bcumulative\b", r"\brolling\b",
        r"\bwindow\b", r"\bover\b",
    ],
    "agg": [
        r"\baverage\b", r"\bmean\b", r"\bsum\b", r"\btotal\b",
        r"\bcount\b", r"\bhow many\b", r"\bnumber of\b",
        r"\bmin(imum)?\b", r"\bmax(imum)?\b",
        r"\bdistinct\b", r"\bunique\b",
        r"\bmedian\b", r"\bstd\b", r"\bvariance\b",
    ],
    "null_nan": [
        r"\bnull\b", r"\bmissing\b", r"\bna\b", r"\bnan\b",
        r"\bfillna\b", r"\bdropna\b", r"\bisna\b", r"\bnotna\b",
    ],
    "concat_stack": [
        r"\bconcat\b", r"\bstack\b", r"\bbind\b",
        r"\bappend\b", r"\bunion\b.*\b(row|frame)\b",
        r"\bvertically\b", r"\bhorizontally\b",
    ],
}


def classify_question(question: str, schema: dict[str, dict[str, str]] | None = None) -> list[str]:
    """
    Return list of relevant API snippet keys for a question.
    Only returns categories where we have targeted snippets — avoids
    injecting noise for things the model already handles via few-shot.
    """
    q = question.lower()
    hits = []

    # String ops — model often uses .startswith() instead of .str.starts_with()
    for pat in _ROUTE_PATTERNS["string"]:
        if re.search(pat, q):
            hits.append("string")
            break

    # Multi-agg groupby — model struggles with list-of-expressions syntax
    is_groupby = any(re.search(p, q) for p in _ROUTE_PATTERNS["groupby"])
    if is_groupby:
        # Check if multiple aggregations are likely requested
        agg_words = ["count", "min", "max", "mean", "average", "sum", "total",
                     "headcount", "number"]
        agg_count = sum(1 for w in agg_words if w in q)
        if agg_count >= 2 or "headcount" in q:
            hits.append("groupby_multi")

    # Anti-join — model doesn't know how="anti"
    negation = any(re.search(p, q) for p in [
        r"\bnever\b", r"\bnot\b", r"\bno\s+\w+s?\b", r"\bwithout\b",
        r"\bwho have not\b", r"\bwho did not\b",
    ])
    if negation and schema and len(schema) > 1:
        hits.append("join_anti")

    # Window functions — rank within group, percentage of total
    for pat in _ROUTE_PATTERNS["window"]:
        if re.search(pat, q):
            hits.append("window")
            break

    # null vs NaN — pandas habits (fillna, isna) vs Polars null/NaN split
    for pat in _ROUTE_PATTERNS["null_nan"]:
        if re.search(pat, q):
            hits.append("null_nan")
            break

    # concat / stack — axis naming differs from pandas
    for pat in _ROUTE_PATTERNS["concat_stack"]:
        if re.search(pat, q):
            hits.append("concat_stack")
            break

    return hits
