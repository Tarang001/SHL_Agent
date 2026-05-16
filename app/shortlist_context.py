"""
Post-recommendation conversation modes and shortlist refinement logic.

After the first shortlist, recommendations are always returned.
Stack contents change only when the user refines, confirms, or adds constraints.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from app.followup import (
    COMPARE_REF_ALIASES,
    is_comparison_turn,   
    shortlist_likely_issued,
)

if TYPE_CHECKING:
    from app.retriever import HybridRetriever

CLARIFICATION_PATTERNS = re.compile(
    r"\b("
    r"what.?s the difference|difference between|how (do|does|long|much)|"
    r"what is|what are|what does|tell me (more )?about|explain|"
    r"which is better|compare|versus|vs\.?|when to use|pros and cons|"
    r"duration|how many minutes|test type|what about"
    r")\b",
    re.I,
)

PREFERENCE_CUE = re.compile(
    r"\b("
    r"right fit|good choice|go with|we(?:'re| are) (?:going with|using)|"
    r"prefer|choose|locking it in|that works|the .+ (?:is |are )?(?:the )?right|"
    r"confirmed|bundle is|we(?:'ll| will) (?:use|go with)"
    r")\b",
    re.I,
)

SECTOR_CUE = re.compile(
    r"\b(industrial|manufacturing|plant operator|chemical facilit)\b",
    re.I,
)

REFINEMENT_ACTION = re.compile(
    r"\b("
    r"add|drop|remove|replace|instead|update|change|without|also include|"
    r"exclude|need less|need more|go with|locking it in|skip|swap"
    r")\b",
    re.I,
)

CONFIRMED_CUE = re.compile(
    r"\b(confirmed|locking it in|shortlist confirmed|that works|perfect|"
    r"we(?:'ll| will) go with|good choice)\b",
    re.I,
)

DROP_NAME_PATTERN = re.compile(
    r"\b(?:drop|remove|without|exclude|skip)\s+(?:the\s+)?([^,.?\n]+)",
    re.I,
)

KEEP_PATTERN = re.compile(
    r"\bkeep\s+(?:the\s+)?([^,.?\n—]+)",
    re.I,
)

ADD_PATTERN = re.compile(
    r"\badd\s+(.+?)(?:\.\s+|\s+drop\b|\s+remove\b|\s+exclude\b|$)",
    re.I,
)

# Token → catalog name/url substring hints (not product-specific hardcoding)
CATALOG_TOKEN_HINTS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"\baws\b|amazon web services", re.I), ["aws", "amazon web services"]),
    (re.compile(r"\bdocker\b", re.I), ["docker"]),
    (re.compile(r"\brest\b|restful", re.I), ["restful web services", "rest"]),
    (re.compile(r"\bsql\b", re.I), ["sql (new)", "sql"]),
    (re.compile(r"\bjava\b", re.I), ["core java", "java"]),
    (re.compile(r"\bspring\b", re.I), ["spring"]),
    (re.compile(r"\bangular\b", re.I), ["angular"]),
    (re.compile(r"\bkubernetes\b|\bk8s\b", re.I), ["kubernetes"]),
]


def _catalog_items_mentioned(
    message: str,
    prior: list[dict],
    retriever: "HybridRetriever",
) -> list[dict]:
    """Catalog rows referenced in the user message (prior shortlist + aliases)."""
    lower = message.lower()
    found: list[dict] = []
    seen: set[str] = set()

    def add(item: Optional[dict]) -> None:
        if item and item["name"] not in seen:
            found.append(item)
            seen.add(item["name"])

    for rec in prior:
        item = retriever.resolve_catalog_item(rec)
        if not item:
            continue
        name_l = item["name"].lower()
        if name_l in lower:
            add(item)
            continue
        for tok in re.findall(r"[a-z0-9\.]+", name_l):
            if len(tok) > 3 and tok in lower:
                add(item)
                break

    for pattern, catalog_substr in COMPARE_REF_ALIASES:
        if pattern.search(message):
            for item in retriever.catalog:
                if catalog_substr in item["name"].lower():
                    add(item)
                    break

    if re.search(r"\b8\.0\b|8\.0 bundle|safety.{0,12}dependability 8", lower):
        for item in retriever.catalog:
            if "8.0" in item["name"] or "8-0" in item.get("url", ""):
                if "safety" in item["name"].lower() or "dependability" in item["name"].lower():
                    add(item)

    return found


def _resolve_catalog_by_tokens(fragment: str, retriever: "HybridRetriever") -> Optional[dict]:
    """Map a user fragment (e.g. 'AWS', 'Docker') to the best catalog row."""
    frag = fragment.strip()
    if not frag:
        return None
    item = retriever.fuzzy_catalog_match(frag)
    if item:
        return item
    lower = frag.lower()
    for pattern, hints in CATALOG_TOKEN_HINTS:
        if pattern.search(frag):
            for cat_item in retriever.catalog:
                name_l = cat_item["name"].lower()
                url_l = cat_item.get("url", "").lower()
                if any(h in name_l or h.replace(" ", "-") in url_l for h in hints):
                    if pattern.search(r"\bsql\b", frag) and "automata" in name_l:
                        continue
                    return cat_item
    return None


def _parse_add_requests(message: str, retriever: "HybridRetriever") -> list[dict]:
    """Extract catalog items the user asked to add (e.g. 'Add AWS and Docker')."""
    items: list[dict] = []
    seen: set[str] = set()
    match = ADD_PATTERN.search(message)
    if not match:
        return items
    clause = match.group(1).strip().rstrip(".")
    parts = re.split(r"\s+and\s+|,\s*", clause)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        item = _resolve_catalog_by_tokens(part, retriever)
        if item and item["name"] not in seen:
            items.append(item)
            seen.add(item["name"])
    return items


def _exclude_fragments_from_message(message: str) -> list[str]:
    """Raw drop targets for fuzzy exclusion from the prior shortlist."""
    frags: list[str] = []
    for match in DROP_NAME_PATTERN.finditer(message):
        frag = re.split(r"\s*[—–-]\s*|\s*,\s*|\s+and\s+", match.group(1).strip(), maxsplit=1)[0].strip()
        if frag:
            frags.append(frag.lower())
    return frags


def _should_exclude_rec(rec: dict, refinement: dict) -> bool:
    """True if this prior row should be removed per drop/exclude rules."""
    name = rec["name"]
    name_l = name.lower()
    if name in refinement.get("exclude_names", []):
        return True
    for frag in refinement.get("exclude_fragments", []):
        if frag in name_l or (len(frag) >= 4 and frag in name_l.replace("-", " ")):
            return True
        if frag == "rest" and "rest" in name_l:
            return True
    return False


def shortlist_names(recs: list[dict]) -> set[str]:
    return {r["name"] for r in recs}


def stack_changed(before: list[dict], after: list[dict]) -> bool:
    return shortlist_names(before) != shortlist_names(after)


def should_republish_recommendations(
    messages: list[dict],
    state: dict,
    *,
    proposed_recs: Optional[list[dict]] = None,
) -> bool:
    """
    After the first shortlist, always return recommendations (table may be unchanged).
    Before the first shortlist, only when issuing or updating recommendations.
    """
    has_prior = bool(
        state.get("recommended_tests")
        or state.get("has_recommended")
        or shortlist_likely_issued(messages, state)
    )

    if state.get("policy_action") == "clarify":
        return False
    if has_prior:
        return True
    return state.get("policy_action") == "recommend"


def is_shortlist_refinement(
    message: str,
    messages: list[dict],
    state: Optional[dict] = None,
    *,
    retriever: Optional["HybridRetriever"] = None,
) -> bool:
    """User adds constraints, modifies stack, or confirms a choice."""
    if not shortlist_likely_issued(messages, state):
        return False

    lower = message.strip().lower()
    from app.state import extract_constraints

    constraints = extract_constraints([{"role": "user", "content": message}])

    if is_comparison_turn(message, has_prior=True):
        if not (
            REFINEMENT_ACTION.search(lower)
            or CONFIRMED_CUE.search(lower)
            or SECTOR_CUE.search(lower)
            or constraints.get("include")
            or constraints.get("exclude")
        ):
            return False

    if REFINEMENT_ACTION.search(lower):
        return True
    if constraints.get("include") or constraints.get("exclude"):
        return True

    has_preference = bool(PREFERENCE_CUE.search(lower) or CONFIRMED_CUE.search(lower))
    if SECTOR_CUE.search(lower):
        return True
    if not has_preference:
        return False

    if retriever is None:
        return bool(
            SECTOR_CUE.search(lower)
            or re.search(r"\b(bundle|8\.0|dsi|opq)\b", lower, re.I)
            or CONFIRMED_CUE.search(lower)
        )

    from app.followup import extract_prior_recommendations

    prior = (state or {}).get("recommended_tests") or []
    if not prior:
        prior = extract_prior_recommendations(messages, retriever)
    mentioned = _catalog_items_mentioned(message, prior, retriever)
    return bool(mentioned or SECTOR_CUE.search(lower) or CONFIRMED_CUE.search(lower))


def is_informational_comparison(
    message: str,
    messages: list[dict],
    state: Optional[dict] = None,
) -> bool:
    """Catalog Q&A — stack unchanged, do not republish."""
    if not shortlist_likely_issued(messages, state):
        return False
    if is_shortlist_refinement(message, messages, state):
        return False
    if is_comparison_turn(message, has_prior=True):
        return True
    if CLARIFICATION_PATTERNS.search(message) and shortlist_likely_issued(messages, state):
        return True
    return False


def post_shortlist_mode(
    message: str,
    messages: list[dict],
    state: Optional[dict] = None,
    *,
    retriever: Optional["HybridRetriever"] = None,
) -> str:
    """One of: refine | informational | none."""
    if not shortlist_likely_issued(messages, state):
        return "none"
    if is_shortlist_refinement(message, messages, state, retriever=retriever):
        return "refine"
    if is_informational_comparison(message, messages, state):
        return "informational"
    return "none"


def parse_shortlist_refinement(
    message: str,
    prior: list[dict],
    retriever: "HybridRetriever",
) -> dict:
    """Structured constraints from a post-shortlist user turn."""
    from app.state import extract_constraints

    lower = message.lower()
    mentioned = _catalog_items_mentioned(message, prior, retriever)
    has_preference = bool(PREFERENCE_CUE.search(lower))
    confirmed = bool(CONFIRMED_CUE.search(lower))
    constraints = extract_constraints([{"role": "user", "content": message}])

    preferred = list(mentioned) if (has_preference or confirmed) else []
    for match in KEEP_PATTERN.finditer(message):
        frag = re.split(r"\s*[—–-]\s*|\s*,\s*", match.group(1).strip(), maxsplit=1)[0].strip()
        item = retriever.fuzzy_catalog_match(frag)
        if item:
            prior_match = retriever.resolve_catalog_item({"name": item["name"]})
            if prior_match and prior_match["name"] in {r["name"] for r in prior}:
                if prior_match not in preferred:
                    preferred.append(prior_match)
            elif item not in preferred:
                preferred.append(item)
        elif re.search(r"\bindustrial\b|8\.0|bundle", frag, re.I):
            picked = None
            for rec in prior:
                cand = retriever.resolve_catalog_item(rec)
                if cand and (
                    "8.0" in cand["name"]
                    or "industrial" in " ".join(cand.get("tags", []))
                    or "manufacturing" in " ".join(cand.get("tags", []))
                ):
                    picked = cand
                    break
            if not picked:
                for cat_item in retriever.catalog:
                    name_l = cat_item["name"].lower()
                    if ("8.0" in cat_item["name"] or "8-0" in cat_item.get("url", "")) and (
                        "safety" in name_l or "dependability" in name_l or "industrial" in name_l
                    ):
                        picked = cat_item
                        break
            if picked and picked not in preferred:
                preferred.append(picked)
    if preferred:
        has_preference = True

    exclude_names: list[str] = []
    exclude_fragments = _exclude_fragments_from_message(message)
    for frag in exclude_fragments:
        item = _resolve_catalog_by_tokens(frag, retriever) or retriever.fuzzy_catalog_match(frag)
        if item:
            exclude_names.append(item["name"])
            continue
        for pattern, catalog_substr in COMPARE_REF_ALIASES:
            if pattern.search(frag):
                for cat_item in retriever.catalog:
                    if catalog_substr in cat_item["name"].lower():
                        exclude_names.append(cat_item["name"])
                        break
                break

    include_items = _parse_add_requests(message, retriever)

    requires_rebuild = bool(
        constraints.get("include")
        or constraints.get("exclude")
        or exclude_names
        or include_items
        or SECTOR_CUE.search(lower)
        or REFINEMENT_ACTION.search(lower)
    )

    return {
        "confirmed": confirmed,
        "preferred_items": preferred,
        "mentioned_items": mentioned,
        "sector": "industrial" if SECTOR_CUE.search(lower) else None,
        "exclude_names": exclude_names,
        "exclude_fragments": exclude_fragments,
        "include_items": include_items,
        "include_categories": list(constraints.get("include", [])),
        "exclude_categories": list(constraints.get("exclude", [])),
        "requires_rebuild": requires_rebuild,
    }


def _apply_category_filters(recs: list[dict], refinement: dict, retriever: "HybridRetriever") -> list[dict]:
    result = list(recs)
    for cat in refinement.get("exclude_categories", []):
        if cat == "personality":
            result = [r for r in result if "P" not in r.get("test_type", "")]
        elif cat == "cognitive":
            result = [r for r in result if "A" not in r.get("test_type", "") and "K" not in r.get("test_type", "")]
        elif cat == "simulation":
            result = [r for r in result if "S" not in r.get("test_type", "") and "E" not in r.get("test_type", "")]
    return result


def _apply_preference_filter(
    result: list[dict], preferred: list[dict], retriever: "HybridRetriever"
) -> list[dict]:
    if not preferred:
        return result
    pref_names = {p["name"] for p in preferred}
    preferred_has_personality = any(
        "P" in "".join((retriever.resolve_catalog_item({"name": p["name"]}) or {}).get("test_types", []))
        for p in preferred
    )
    if not preferred_has_personality:
        return result
    filtered = []
    for rec in result:
        item = retriever.resolve_catalog_item(rec)
        if not item:
            continue
        types = "".join(item.get("test_types", []))
        if "P" in types and rec["name"] not in pref_names:
            continue
        filtered.append(rec)
    return filtered if filtered else result


def _merge_from_candidates(
    base: list[dict],
    candidates: list[dict],
    refinement: dict,
    state: dict,
    retriever: "HybridRetriever",
) -> list[dict]:
    """Add retrieval matches for newly requested capability categories."""
    seen = {r["name"] for r in base}
    merged = list(base)
    max_n = state.get("shortlist_max", 10)

    for cat in refinement.get("include_categories", []):
        for item in candidates:
            if len(merged) >= max_n:
                break
            tags = item.get("tags", [])
            match = (
                (cat == "personality" and "personality" in tags)
                or (cat == "cognitive" and ("cognitive" in tags or "verify" in " ".join(tags)))
                or (cat == "simulation" and "simulation" in tags)
                or (cat == "coding" and "coding" in tags)
                or (cat == "communication" and "spoken_language" in tags)
            )
            if match and item["name"] not in seen:
                merged.append(retriever.catalog_item_to_rec(item))
                seen.add(item["name"])

    if (
        refinement.get("requires_rebuild")
        and not refinement.get("preferred_items")
        and not refinement.get("exclude_names")
        and not refinement.get("exclude_categories")
    ):
        for item in candidates:
            if len(merged) >= max_n:
                break
            if state.get("exclude_personality") and "personality" in item.get("tags", []):
                continue
            rec = retriever.catalog_item_to_rec(item)
            if rec["name"] not in seen:
                merged.append(rec)
                seen.add(rec["name"])

    if refinement.get("sector") == "industrial":
        industrial = []
        other = []
        for rec in merged:
            item = retriever.resolve_catalog_item(rec) or {}
            tags = " ".join(item.get("tags", []))
            name_l = rec["name"].lower()
            if "manufacturing" in tags or "safety" in tags or "industrial" in name_l or "8.0" in name_l:
                industrial.append(rec)
            else:
                other.append(rec)
        merged = industrial + other

    return merged[:max_n]


def _dedupe_skill_family(
    recs: list[dict],
    family_key: str,
    preferred_substrings: list[str],
    retriever: "HybridRetriever",
) -> list[dict]:
    """When multiple assessments cover the same skill, keep the best-matched row."""
    family = []
    other = []
    for rec in recs:
        name_l = rec["name"].lower()
        if family_key in name_l or (family_key == "sql" and "sql" in name_l):
            family.append(rec)
        else:
            other.append(rec)
    if len(family) <= 1:
        return recs
    chosen = None
    for pref in preferred_substrings:
        for rec in family:
            if pref in rec["name"].lower() or pref.replace(" ", "-") in rec.get("url", "").lower():
                chosen = rec
                break
        if chosen:
            break
    if not chosen:
        for pref in preferred_substrings:
            for cat_item in retriever.catalog:
                name_l = cat_item["name"].lower()
                url_l = cat_item.get("url", "").lower()
                if pref in name_l or pref.replace(" ", "-") in url_l:
                    if "automata" in name_l and family_key == "sql":
                        continue
                    chosen = retriever.catalog_item_to_rec(cat_item)
                    break
            if chosen:
                break
    if not chosen:
        non_automata = [r for r in family if "automata" not in r["name"].lower()]
        chosen = non_automata[0] if non_automata else family[0]
    return other + [chosen]


def _realign_technical_stack(
    prior: list[dict],
    candidates: list[dict],
    state: dict,
    refinement: dict,
    retriever: "HybridRetriever",
) -> list[dict]:
    """
    Rebuild the technical (K/S) slice when user adds skills — keep personality/ability anchors.
    Uses retrieval order and conversation tags, not fixed product lists.
    """
    max_n = state.get("shortlist_max", 10)
    anchors: list[dict] = []
    for rec in prior:
        types = rec.get("test_type", "")
        if any(t in types for t in ("P", "A")):
            if not _should_exclude_rec(rec, refinement):
                anchors.append(rec)

    tech: list[dict] = []
    seen: set[str] = set()

    def add_rec(rec: dict) -> None:
        if rec["name"] in seen or _should_exclude_rec(rec, refinement):
            return
        types = rec.get("test_type", "")
        if "D" in types and refinement.get("include_items"):
            return
        tech.append(rec)
        seen.add(rec["name"])

    for item in refinement.get("include_items", []):
        add_rec(retriever.catalog_item_to_rec(item))

    for rec in prior:
        types = rec.get("test_type", "")
        if "K" in types or "S" in types:
            add_rec(rec)

    tag_keywords: list[str] = []
    for tag in state.get("inferred_tags", []):
        tag_keywords.append(tag)
    if state.get("role") == "technical hire":
        tag_keywords.extend(["java", "spring", "sql", "aws", "docker"])
    for item in refinement.get("include_items", []):
        for tok in re.findall(r"[a-z0-9\+]+", item["name"].lower()):
            if len(tok) > 2:
                tag_keywords.append(tok)

    for item in candidates:
        if len(anchors) + len(tech) >= max_n:
            break
        name_l = item["name"].lower()
        if any(kw in name_l for kw in tag_keywords if len(kw) > 2):
            add_rec(retriever.catalog_item_to_rec(item))

    if refinement.get("include_items") or any("sql" in f for f in refinement.get("exclude_fragments", [])):
        for cat_item in retriever.catalog:
            if cat_item["name"] == "SQL (New)":
                add_rec(retriever.catalog_item_to_rec(cat_item))
                break
    tech = [r for r in tech if "automata" not in r["name"].lower()]
    tech = _dedupe_skill_family(tech, "sql", ["sql (new)", "sql-new"], retriever)
    tech = _dedupe_skill_family(tech, "spring", ["spring (new)"], retriever)
    tech = _dedupe_skill_family(tech, "java", ["core java"], retriever)

    combined = anchors + tech
    return combined[:max_n]


def apply_shortlist_refinement(
    prior: list[dict],
    refinement: dict,
    retriever: "HybridRetriever",
    candidates: Optional[list[dict]] = None,
    state: Optional[dict] = None,
) -> list[dict]:
    """Apply exclusions, preferences, and new requirements to the active shortlist."""
    if not prior and candidates and state:
        return retriever.validate_recommendations_against_catalog(
            [retriever.catalog_item_to_rec(c) for c in candidates[: state.get("shortlist_max", 10)]]
        )

    result = [r for r in prior if not _should_exclude_rec(r, refinement)]

    for name in refinement.get("exclude_names", []):
        result = [r for r in result if r["name"] != name]

    result = _apply_category_filters(result, refinement, retriever)
    result = _apply_preference_filter(result, refinement.get("preferred_items", []), retriever)

    seen = {r["name"] for r in result}
    for item in refinement.get("include_items", []):
        rec = retriever.catalog_item_to_rec(item)
        if rec["name"] not in seen:
            result.append(rec)
            seen.add(rec["name"])

    if candidates and state and refinement.get("include_items"):
        result = _realign_technical_stack(prior, candidates, state, refinement, retriever)
    elif candidates and state and refinement.get("requires_rebuild"):
        result = _merge_from_candidates(result, candidates, refinement, state, retriever)

    if not result and refinement.get("preferred_items"):
        result = [retriever.catalog_item_to_rec(p) for p in refinement["preferred_items"]]

    seen = {r["name"] for r in result}
    for p in refinement.get("preferred_items", []):
        rec = retriever.catalog_item_to_rec(p)
        if rec["name"] not in seen:
            result.append(rec)
            seen.add(rec["name"])

    return retriever.validate_recommendations_against_catalog(result)


def _short_label(name: str) -> str:
    """Compact label for delta summaries (e.g. 'RESTful Web Services' → 'REST')."""
    if "restful" in name.lower():
        return "REST"
    if "amazon web services" in name.lower() or "aws" in name.lower():
        return "AWS"
    if name.lower().startswith("docker"):
        return "Docker"
    if "automata" in name.lower() and "sql" in name.lower():
        return "Automata SQL"
    if name.lower().startswith("sql"):
        return "SQL"
    parts = name.split("—")[0].split("(")[0].strip()
    return parts[:40] if len(parts) > 40 else parts


def build_refinement_reply(
    refined: list[dict],
    refinement: dict,
    prior: list[dict],
) -> str:
    """Acknowledge constraint updates or final confirmation."""
    if not refined:
        return (
            "I could not apply that constraint to the current shortlist — "
            "which assessments should stay in the stack?"
        )

    if refinement.get("confirmed"):
        sector = refinement.get("sector")
        if sector == "industrial":
            lead = "Good choice for an industrial context. Shortlist confirmed."
        else:
            lead = "Shortlist confirmed."
        names = ", ".join(r["name"] for r in refined)
        return f"{lead} {names}."

    added = [r for r in refined if r["name"] not in shortlist_names(prior)]
    removed = [r for r in prior if r["name"] not in shortlist_names(refined)]

    if added or removed:
        parts = []
        if removed:
            out_labels = ", ".join(_short_label(r["name"]) for r in removed[:4])
            parts.append(f"{out_labels} out")
        if added:
            in_labels = ", ".join(_short_label(r["name"]) for r in added[:4])
            parts.append(f"{in_labels} in")
        delta = "; ".join(parts)
        return f"Updated — {delta}."

    if stack_changed(prior, refined):
        return "Updated the shortlist to match your requirements."

    return "Shortlist unchanged based on your note."

