"""
Post-recommendation follow-up handling — memory, intent detection, grounded comparisons.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.retriever import HybridRetriever

COMPARISON_TURN_PATTERNS = re.compile(
    r"\b("
    r"difference between|what.?s the difference|compare|comparison|vs\.?|versus|"
    r"which is better|how (do|does) .+ differ|better for"
    r")\b",
    re.I,
)

SHORTLIST_ISSUED_PATTERN = re.compile(
    r"\b("
    r"here are recommended|recommended assessments|your shortlist|"
    r"based on your (query|requirements)|these shl assessments|best fit"
    r")\b",
    re.I,
)

FOLLOWUP_PATTERNS = re.compile(
    r"\b("
    r"difference between|what.?s the difference|compare|comparison|vs\.?|versus|"
    r"which is better|how (do|does) .+ differ|better for|"
    r"suitab(le|ility)|should (we|i) (use|include|add|keep)|"
    r"replace|alternative|instead of|explain|why (both|these|did you)|"
    r"pros and cons|when to use|trade.?off|"
    r"do we (still )?need|can i (drop|remove|swap)|"
    r"what is|what are|how long|duration|tell me (more )?about|"
    r"leadership vs|personality vs|cognitive vs|behavioral vs|sales vs|"
    r"worth (it|using)|what about|how (is|are)"
    r")\b",
    re.I,
)

# Longer phrases first — avoid "opq" swallowing "opq mq sales"
COMPARE_REF_ALIASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"opq\s*mq\s*sales(?:\s*report)?", re.I), "opq mq sales report"),
    (re.compile(r"global\s+skills\s+development(?:\s*report)?", re.I), "global skills development report"),
    (re.compile(r"global\s+skills\s+assessment", re.I), "global skills assessment"),
    (re.compile(r"sales\s+transformation", re.I), "sales transformation 2.0"),
    (re.compile(r"opq32?r?", re.I), "occupational personality questionnaire opq32r"),
    (re.compile(r"\bopq\b", re.I), "occupational personality questionnaire opq32r"),
    (re.compile(r"\bgsa\b", re.I), "global skills assessment"),
    (re.compile(r"\bgsi\b", re.I), "global skills"),
    (re.compile(r"\bdsi\b", re.I), "dependability and safety instrument"),
    (re.compile(r"verify\s*g\+?", re.I), "shl verify interactive g"),
    (re.compile(r"\bsvar\b", re.I), "svar"),
]


def shortlist_likely_issued(messages: list[dict], state: Optional[dict] = None) -> bool:
    """True when a prior assistant turn delivered a shortlist (structured or prose-only)."""
    if state and (state.get("has_recommendations") or state.get("recommended_tests")):
        return True
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        if msg.get("recommendations"):
            return True
        content = msg.get("content") or ""
        if SHORTLIST_ISSUED_PATTERN.search(content):
            return True
        if len(re.findall(r"https://www\.shl\.com/\S+", content)) >= 2:
            return True
    return False


def rebuild_shortlist_from_context(
    messages: list[dict],
    state: dict,
    retriever: "HybridRetriever",
) -> list[dict]:
    """
    Recover the prior shortlist from structured history, assistant prose, or retrieval.
    Used when the client did not echo recommendations on prior assistant turns.
    """
    prior = extract_prior_recommendations(messages, retriever)
    if prior:
        return prior

    if not shortlist_likely_issued(messages, state):
        return []

    from app.state import build_search_query
    from app.conversation_policy import shortlist_bounds

    query = build_search_query(state, messages)
    candidates = retriever.retrieve(query, state, top_k=40)
    _min, max_n = shortlist_bounds(messages, state)
    recs: list[dict] = []
    for item in candidates:
        if state.get("exclude_personality") and "personality" in item.get("tags", []):
            continue
        recs.append(retriever.catalog_item_to_rec(item))
        if len(recs) >= max_n:
            break
    return retriever.validate_recommendations_against_catalog(recs)


def extract_prior_recommendations(
    messages: list[dict], retriever: "HybridRetriever"
) -> list[dict]:
    """
    Return the most recent published shortlist (not a union of every past turn).
    Informational turns with recommendations: [] are skipped so the active stack persists.
    """
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue

        recs_raw = msg.get("recommendations")
        if isinstance(recs_raw, list):
            if recs_raw:
                found = []
                seen: set[str] = set()
                for rec in recs_raw:
                    if not isinstance(rec, dict):
                        continue
                    item = retriever.resolve_catalog_item(rec)
                    if item and item["name"] not in seen:
                        found.append(retriever.catalog_item_to_rec(item))
                        seen.add(item["name"])
                if found:
                    return retriever.validate_recommendations_against_catalog(found)
            continue

        content = msg.get("content", "")
        if len(re.findall(r"https://www\.shl\.com/\S+", content)) >= 2:
            found = retriever.recommendations_from_assistant_text(content)
            if found:
                return retriever.validate_recommendations_against_catalog(found)

    return []


def is_comparison_turn(user_message: str, *, has_prior: bool = False) -> bool:
    """Explicit comparison only — not clarification slot fills like 'US' or 'English'."""
    from app.intent import is_explicit_comparison

    return is_explicit_comparison(user_message, has_prior=has_prior)


def is_followup_query(
    user_message: str,
    prior_recommendations: list[dict],
    *,
    has_recommended: bool = False,
) -> bool:
    """True when user is asking about an existing shortlist, not requesting a new one."""
    if not has_recommended and not prior_recommendations:
        return False
    if is_comparison_turn(user_message, has_prior=has_recommended):
        return True
    if not prior_recommendations:
        return False
    if FOLLOWUP_PATTERNS.search(user_message):
        return True
    # Mentions a prior assessment by name or shorthand
    lower = user_message.lower()
    for rec in prior_recommendations:
        name_lower = rec["name"].lower()
        if name_lower in lower:
            return True
        for token in re.findall(r"[a-z0-9\+]{4,}", name_lower):
            if token in lower and len(token) > 4:
                return True
    from app.intent import is_clarification_fill

    if is_clarification_fill(user_message):
        return False
    return False


def _normalize_comparison_side(side: str) -> str:
    """Strip question framing so catalog fuzzy-match sees product names."""
    s = side.strip(" ?.")
    s = re.sub(r"^(?:is|are)\s+(?:the\s+)?", "", s, flags=re.I)
    s = re.sub(r"^the\s+", "", s, flags=re.I)
    return s.strip()


def _extract_comparison_sides(query: str) -> list[str]:
    """Split 'difference between X and Y' / 'X vs Y' into two sides."""
    patterns = [
        r"difference between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"(?:is|are)\s+(?:the\s+)?(.+?)\s+different from\s+(?:the\s+)?(.+?)(?:\?|$)",
        r"(.+?)\s+different from\s+(?:the\s+)?(.+?)(?:\?|$)",
        r"compare\s+(.+?)\s+(?:vs\.?|versus|and)\s+(.+?)(?:\?|$)",
        r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)",
    ]
    for pat in patterns:
        m = re.search(pat, query.strip(), re.I)
        if m:
            return [
                _normalize_comparison_side(m.group(1)),
                _normalize_comparison_side(m.group(2)),
            ]
    return []


def _resolve_side_to_catalog(
    side: str,
    retriever: "HybridRetriever",
    prior_recommendations: Optional[list[dict]] = None,
) -> Optional[dict]:
    side_clean = side.strip()
    if not side_clean:
        return None

    for pattern, catalog_substr in COMPARE_REF_ALIASES:
        if pattern.search(side_clean):
            for item in retriever.catalog:
                if catalog_substr in item["name"].lower():
                    return item

    for rec in prior_recommendations or []:
        item = retriever.resolve_catalog_item(rec)
        if not item:
            continue
        name_lower = item["name"].lower()
        side_lower = side_clean.lower()
        if side_lower in name_lower or name_lower in side_lower:
            return item

    return retriever.fuzzy_catalog_match(side_clean)


def resolve_compared_catalog_items(
    query: str,
    retriever: "HybridRetriever",
    prior_recommendations: Optional[list[dict]] = None,
    candidates: Optional[list[dict]] = None,
) -> list[dict]:
    """Find catalog rows referenced in a comparison / follow-up question."""
    matched: list[dict] = []
    seen: set[str] = set()

    def add_item(item: Optional[dict]) -> None:
        if item and item["name"] not in seen:
            matched.append(item)
            seen.add(item["name"])

    sides = _extract_comparison_sides(query)
    if sides:
        for side in sides:
            add_item(_resolve_side_to_catalog(side, retriever, prior_recommendations))

    if len(matched) >= 2:
        return matched[:4]

    q = query.lower()
    for pattern, catalog_substr in COMPARE_REF_ALIASES:
        if pattern.search(query):
            for item in retriever.catalog:
                if catalog_substr in item["name"].lower():
                    add_item(item)

    for rec in prior_recommendations or []:
        item = retriever.resolve_catalog_item(rec)
        if item:
            name_lower = item["name"].lower()
            if name_lower in q or any(
                tok in name_lower
                for tok in re.findall(r"[a-z0-9\+]+", q)
                if len(tok) > 3
            ):
                add_item(item)

    for item in candidates or []:
        name_lower = item["name"].lower()
        if name_lower in q:
            add_item(item)
        elif any(tok in name_lower for tok in re.findall(r"[a-z0-9\+]+", q) if len(tok) > 3):
            add_item(item)

    if prior_recommendations and len(matched) < 2:
        for rec in prior_recommendations:
            add_item(retriever.resolve_catalog_item(rec))
            if len(matched) >= 2:
                break

    return matched[:4]


def build_deterministic_comparison_reply(items: list[dict]) -> str:
    """Grounded comparison prose from catalog fields only."""
    if not items:
        return (
            "I can compare assessments from your shortlist or the SHL catalog — "
            "which two products would you like contrasted?"
        )
    if len(items) == 1:
        item = items[0]
        labels = ", ".join(item.get("test_type_labels", item.get("test_types", [])))
        desc = (item.get("description") or "").strip()
        if len(desc) > 280:
            desc = desc[:277].rstrip() + "..."
        return (
            f"{item['name']} is catalogued as {labels}. "
            f"{desc}"
        )

    summaries = []
    for item in items[:2]:
        labels = ", ".join(item.get("test_type_labels", item.get("test_types", [])))
        desc = (item.get("description") or "").strip()
        if len(desc) > 240:
            desc = desc[:237].rstrip() + "..."
        summaries.append(f"{item['name']} ({labels}): {desc}")

    a_name = items[0]["name"]
    b_name = items[1]["name"]
    return (
        f"Yes — they are different assessments.\n\n"
        f"{summaries[0]}\n\n"
        f"{summaries[1]}\n\n"
        f"In practice, {a_name} and {b_name} differ in scope, format, and intended use case "
        f"as described above."
    )


def format_comparison_context(items: list[dict]) -> str:
    blocks = []
    for item in items:
        types = ",".join(item.get("test_types", []))
        labels = ", ".join(item.get("test_type_labels", [])[:3])
        levels = ", ".join(item.get("job_levels", [])[:4]) or "multiple levels"
        blocks.append(
            f'COMPARE: "{item["name"]}"\n'
            f"  types:{types} ({labels})\n"
            f"  target_levels:{levels}\n"
            f"  purpose_tags:{', '.join(item.get('tags', [])[:6])}\n"
            f"  url:{item['url']}\n"
            f"  summary:{item.get('description', '')[:280]}"
        )
    return "\n".join(blocks)

followup.py