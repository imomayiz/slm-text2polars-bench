"""
Shared prompt routing: targeted API snippets + classify_question(...).

Used by inference.py and generator.py so the CUDA bench path does not need generator.py.
"""

from __future__ import annotations

import re
from typing import Any

# Only inject snippets for tricky APIs where small models commonly fail.
# Simple categories (filter, agg, sort) are already well-covered by few-shot examples.
API_SNIPPETS = {
    "string": (
        "# Polars uses underscores: .str.starts_with() .str.ends_with() .str.contains()\n"
        "# NOT .startswith() or .endswith()"
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

    return hits
