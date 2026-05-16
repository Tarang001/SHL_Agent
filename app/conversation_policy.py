"""
Conversation policy derived from gold hiring-consultant dialogues.

Principles (not phrase matching):
- Recommend only when the scenario's decision-critical slots are filled, the user
  explicitly approves a proposed stack, or a well-specified single-turn brief exists.
- Broad role labels (leadership, engineers, support) always need narrowing first.
- Multi-turn JD / executive / contact-center flows may need 2–3 clarifications before recs.
- Comparisons and legal refusals stay recommendation-free unless refining an existing stack.
"""
from __future__ import annotations

import re
from typing import Optional

# ── Archetypes: each defines slots that must be filled before first recommendation ──

ARCHETYPE_SLOTS: dict[str, list[str]] = {
    "leadership_executive": ["audience", "hiring_purpose"],
    "contact_center": ["operating_context", "spoken_language", "english_variant"],
    "technical_jd": ["stack_focus", "role_model"],
    "healthcare_bilingual": ["language_strategy"],
    "catalog_confirm": ["user_go_ahead"],
    "general_broad": ["role_specificity", "competency_or_purpose"],
}

# Rich single-turn briefs where the PDF recommends immediately (no extra slots)
IMMEDIATE_ARCHETYPES = frozenset({
    "safety_plant",
    "graduate_battery",
    "sales_audit",
    "admin_screening",
})

# Target shortlist size (min, max) learned from gold traces — not a fixed count
SHORTLIST_BOUNDS: dict[str, tuple[int, int]] = {
    "leadership_executive": (2, 4),
    "contact_center": (3, 4),
    "technical_jd": (4, 8),
    "safety_plant": (2, 4),
    "graduate_battery": (2, 4),
    "sales_audit": (4, 5),
    "admin_screening": (2, 5),
    "healthcare_bilingual": (4, 5),
    "catalog_confirm": (3, 6),
    "general_broad": (3, 7),
}

USER_GO_AHEAD_PATTERN = re.compile(
    r"\b("
    r"yes,?\s*go ahead|yes go ahead|go ahead|build (the |a )?shortlist|"
    r"go with (the )?hybrid|confirmed|locking it in|keep the shortlist|"
    r"that works|perfect|understood\.?\s*keep"
    r")\b",
    re.I,
)

USER_DONE_PATTERN = re.compile(
    r"\b("
    r"perfect|confirmed|that'?s what we need|that works|thanks|locking it in|"
    r"keep the shortlist|good (choice|two-stage)|clear\.\s*we"
    r")\b",
    re.I,
)

BROAD_CATEGORY_PATTERN = re.compile(
    r"\b("
    r"senior leadership|leadership (assessments?|solutions?)?|solution for .{0,20}leadership|"
    r"software engineers?|developers?|freshers?|graduate hires?|"
    r"customer support|operations|sales hiring|need assessments for managers?|"
    r"hiring for operations|hiring software engineers?"
    r")\b",
    re.I,
)

JD_PATTERN = re.compile(
    r"\b(job description|here.?s the jd|across core java|microservice|5\+ years)\b",
    re.I,
)

COMPARISON_PATTERNS = re.compile(
    r"\b(difference between|compare|comparison|vs\.?|versus|which is better|"
    r"how (do|does) .+ differ|what.?s the difference)\b",
    re.I,
)

REFINEMENT_PATTERNS = re.compile(
    r"\b(add|drop|remove|replace|instead|update|change|without|also include|keep|"
    r"actually|need less|need more|include|exclude|locking it in)\b",
    re.I,
)

# Explicit request to (re-)produce a shortlist after one was already given
RE_RECOMMEND_PATTERNS = re.compile(
    r"\b("
    r"recommend|shortlist|suggest (some |an )?assessments?|build (the |a )?shortlist|"
    r"show (me )?(the )?(tests?|assessments?|stack)|give me (the )?tests?|"
    r"what (tests?|assessments?) should|update (the )?list|refresh (the )?stack|"
    r"new stack|revise (the )?stack"
    r")\b",
    re.I,
)

# Follow-up Q&A about tests already discussed — answer in prose only
INFO_FOLLOWUP_PATTERNS = re.compile(
    r"\b("
    r"what is|what are|how long|how much time|duration|tell me (more )?about|explain|"
    r"why (did you|do you)|can you describe|what does|how does|"
    r"which (one|test|assessment)|is (this|it|that)|are (these|they)|"
    r"more (info|information|detail)|what about|how (is|are)|"
    r"worth (it|using)|should (we|i) (use|include)|pros and cons"
    r")\b",
    re.I,
)

OFFTOPIC_LEGAL = re.compile(
    r"\b(legally required|legal obligation|regulatory requirement|"
    r"does this .+ satisfy)\b",
    re.I,
)


def _is_comparison(messages: list[dict], state: dict) -> bool:
    if state.get("conversation_intent") == "COMPARISON_REQUEST":
        return True
    if state.get("conversation_goal") == "compare":
        return True
    from app.intent import is_explicit_comparison

    has_prior = bool(state.get("has_recommended") or state.get("recommended_tests"))
    return is_explicit_comparison(_last(messages), has_prior=has_prior)


def _is_refinement(messages: list[dict], state: dict) -> bool:
    if state.get("conversation_goal") == "refine":
        return True
    if not state.get("has_recommended"):
        return False
    return bool(REFINEMENT_PATTERNS.search(_last(messages)))


def _is_offtopic_last(messages: list[dict]) -> bool:
    return bool(OFFTOPIC_LEGAL.search(_last(messages)))


def _text(messages: list[dict]) -> str:
    return " ".join(m["content"] for m in messages if m["role"] == "user").lower()


def _last(messages: list[dict]) -> str:
    return messages[-1]["content"].lower()


def _turns(messages: list[dict]) -> int:
    return len([m for m in messages if m["role"] == "user"])


def infer_archetype(messages: list[dict], state: dict) -> str:
    """Classify the hiring conversation type from cumulative user text."""
    text = _text(messages)
    tags = set(state.get("inferred_tags", []))

    if re.search(r"\b(plant operator|chemical facilit).{0,80}(safety|dependability|procedure)\b", text, re.I):
        return "safety_plant"
    if re.search(
        r"\b(graduate management trainee|trainee scheme).{0,60}"
        r"(cognitive|personality|sjt|situational|battery)\b",
        text,
        re.I,
    ):
        return "graduate_battery"
    if re.search(r"\b(re-?skill|talent audit|restructur).{0,40}sales\b", text, re.I) or (
        "reskilling" in tags and "sales" in tags
    ):
        return "sales_audit"
    if re.search(r"\b(screen|excel|word).{0,40}(admin|assistant)\b", text, re.I) or (
        "microsoft_office" in tags and re.search(r"\bexcel\b.*\bword\b", text, re.I)
    ):
        return "admin_screening"
    if re.search(r"\b(bilingual|spanish).{0,60}(healthcare|hipaa|patient records)\b", text, re.I):
        return "healthcare_bilingual"
    if re.search(r"\b(rust)\b", text, re.I) and re.search(
        r"\bwhat assessments should\b", text, re.I
    ):
        return "catalog_confirm"
    if re.search(r"\b(contact cent|call cent).{0,40}(agent|inbound|screening)\b", text, re.I):
        return "contact_center"
    if re.search(r"\b(senior leadership|solution for leadership|cxo|director.level)\b", text, re.I) or (
        "leadership" in tags and re.search(r"\bleadership\b", text, re.I)
    ):
        return "leadership_executive"
    if JD_PATTERN.search(text) or (
        len(text.split()) >= 40 and re.search(r"\b(java|spring|angular|docker|aws)\b", text, re.I)
    ):
        return "technical_jd"
    if BROAD_CATEGORY_PATTERN.search(text):
        return "general_broad"
    if re.search(r"\b(engineer|developer|analyst|manager|support|sales)\b", text, re.I):
        return "general_broad"
    return "general_broad"


def _slot_audience(text: str) -> bool:
    return bool(re.search(
        r"\b(cxo|ceo|cto|cfo|director|vp |vice president|c-suite|15\+|15 plus|"
        r"director.level|executive pool|senior leadership pool)\b",
        text,
        re.I,
    ))


def _slot_hiring_purpose(text: str) -> bool:
    return bool(re.search(
        r"\b(selection|developmental|development feedback|succession|benchmark|"
        r"comparing candidates|newly created position|already in role|promotion)\b",
        text,
        re.I,
    ))


def _slot_operating_context(text: str) -> bool:
    return bool(re.search(
        r"\b(entry.level|contact cent|call cent|inbound|customer service|500|screening)\b",
        text,
        re.I,
    ))


def _slot_spoken_language(text: str) -> bool:
    return bool(re.search(
        r"\b(english|spanish|french|german|portuguese|bilingual)\b",
        text,
        re.I,
    ))


def _slot_english_variant(text: str) -> bool:
    if not re.search(r"\benglish\b", text, re.I):
        return bool(re.search(r"\b(us|uk|australian|indian accent)\b", text, re.I))
    return bool(re.search(
        r"\b(us|usa|uk|british|australian|indian accent)\b|"
        r"english\s*\(\s*us\s*\)",
        text,
        re.I,
    ))


def _slot_stack_focus(text: str) -> bool:
    return bool(re.search(
        r"\b(backend.leaning|frontend.heavy|balanced full.stack|backend.heavy|"
        r"primary.{0,20}(java|spring)|angular.{0,30}(occasional|secondary|review))\b",
        text,
        re.I,
    ))


def _slot_role_model(text: str) -> bool:
    return bool(re.search(
        r"\b(senior ic|tech lead|don.?t manage|own services|lead design on|"
        r"architecture across|individual contributor)\b",
        text,
        re.I,
    ))


def _slot_language_strategy(text: str) -> bool:
    return bool(re.search(
        r"\b(hybrid|personality.only|go with the hybrid|english fluent for written)\b",
        text,
        re.I,
    ))


def _slot_user_go_ahead(text: str) -> bool:
    return bool(USER_GO_AHEAD_PATTERN.search(text))


def _slot_role_specificity(text: str, state: dict) -> bool:
    tags = set(state.get("inferred_tags", []))
    if re.search(
        r"\b(java|python|rust|spring|sql|plant operator|hipaa|excel|word|"
        r"inside sales|data analyst)\b",
        text,
        re.I,
    ):
        return True
    return bool(tags.intersection({
        "java", "python", "rust", "sql", "safety", "hipaa", "microsoft_office", "contact_center",
    }))


def _slot_competency_or_purpose(text: str, state: dict) -> bool:
    if re.search(
        r"\b(stakeholder|empathy|spoken english|safety|dependability|procedure|"
        r"cognitive|personality|sjt|inbound|benchmark|audit|screening)\b",
        text,
        re.I,
    ):
        return True
    return bool(state.get("include_cognitive") or state.get("include_personality"))


def filled_slots(archetype: str, messages: list[dict], state: dict) -> dict[str, bool]:
    text = _text(messages)
    checks = {
        "audience": _slot_audience(text),
        "hiring_purpose": _slot_hiring_purpose(text),
        "operating_context": _slot_operating_context(text),
        "spoken_language": _slot_spoken_language(text),
        "english_variant": _slot_english_variant(text),
        "stack_focus": _slot_stack_focus(text),
        "role_model": _slot_role_model(text),
        "language_strategy": _slot_language_strategy(text),
        "user_go_ahead": _slot_user_go_ahead(text),
        "role_specificity": _slot_role_specificity(text, state),
        "competency_or_purpose": _slot_competency_or_purpose(text, state),
    }
    required = ARCHETYPE_SLOTS.get(archetype, ARCHETYPE_SLOTS["general_broad"])
    return {slot: checks.get(slot, False) for slot in required}


def missing_slots(archetype: str, messages: list[dict], state: dict) -> list[str]:
    if archetype in IMMEDIATE_ARCHETYPES:
        return []
    filled = filled_slots(archetype, messages, state)
    return [slot for slot, ok in filled.items() if not ok]


def conversation_phase(messages: list[dict], state: dict) -> str:
    """
    discovery → narrowing → proposing → recommending → refining → comparing → closing
    """
    if _is_offtopic_last(messages):
        return "refuse"
    if _is_refinement(messages, state):
        return "refining"
    if _is_comparison(messages, state):
        return "comparing"
    if user_confirmed_done(messages, state):
        return "closing"

    archetype = infer_archetype(messages, state)
    missing = missing_slots(archetype, messages, state)

    if archetype == "catalog_confirm" and missing:
        return "proposing"
    if missing:
        if archetype == "leadership_executive" and _slot_audience(_text(messages)) and not _slot_hiring_purpose(
            _text(messages)
        ):
            return "narrowing"
        if archetype in IMMEDIATE_ARCHETYPES:
            return "recommending"
        return "narrowing" if _turns(messages) > 1 else "discovery"

    if should_recommend_slots(messages, state):
        return "recommending"
    return "discovery"


def should_recommend_slots(messages: list[dict], state: dict) -> bool:
    """Core recommend gate aligned with gold dialogue structure."""
    from app.intent import FORCE_RECOMMEND_TURN, has_sufficient_context

    if _turns(messages) >= FORCE_RECOMMEND_TURN:
        return True
    if state.get("conversation_intent") == "SUFFICIENT_FOR_RECOMMENDATION":
        return True
    if state.get("conversation_intent") == "CLARIFICATION_RESPONSE" and has_sufficient_context(messages, state):
        return True
    if _is_refinement(messages, state):
        return True

    text = _text(messages)
    archetype = infer_archetype(messages, state)

    if USER_GO_AHEAD_PATTERN.search(text) and _turns(messages) >= 1:
        if archetype != "general_broad" or _slot_role_specificity(text, state):
            return True

    if archetype in IMMEDIATE_ARCHETYPES:
        return True

    missing = missing_slots(archetype, messages, state)
    if missing:
        return False

    if archetype == "general_broad":
        filled = filled_slots(archetype, messages, state)
        return sum(filled.values()) >= 2

    return True


def should_clarify_slots(messages: list[dict], state: dict) -> bool:
    from app.intent import FORCE_RECOMMEND_TURN

    if _turns(messages) >= FORCE_RECOMMEND_TURN:
        return False
    if state.get("conversation_intent") in (
        "SUFFICIENT_FOR_RECOMMENDATION",
        "COMPARISON_REQUEST",
        "FOLLOWUP_REASONING",
        "REFINEMENT_REQUEST",
    ):
        return False
    if _is_offtopic_last(messages):
        return False
    if _is_refinement(messages, state):
        return False
    if _is_comparison(messages, state) and not state.get("has_recommended"):
        return False
    return not should_recommend_slots(messages, state)


def should_emit_recommendations(messages: list[dict], state: dict) -> bool:
    """
    Whether the response should include a recommendations array.
    Delegates to the core republish rule in shortlist_context.
    """
    from app.shortlist_context import should_republish_recommendations

    if state.get("conversation_intent") == "VAGUE_INITIAL_QUERY":
        return False
    if should_clarify_slots(messages, state):
        return False
    if _is_offtopic_last(messages):
        return False
    return should_republish_recommendations(messages, state)


def shortlist_bounds(messages: list[dict], state: dict) -> tuple[int, int]:
    """
    How many recommendations to return (min, max) for this scenario.
    Varies by archetype and user constraints — not a fixed bundle size.
    """
    archetype = infer_archetype(messages, state)
    min_n, max_n = SHORTLIST_BOUNDS.get(archetype, (3, 7))
    text = _text(messages)

    if re.search(r"\b(quick|fast|rapid|lean|minimal|lightweight)\b", text, re.I):
        max_n = min(max_n, 4)
        min_n = min(min_n, 2)
    if re.search(r"\b(full battery|comprehensive|complete stack|all three)\b", text, re.I):
        max_n = min(10, max_n + 1)
    if re.search(r"\b(only|just)\s+(two|2|three|3)\b", text, re.I):
        m = re.search(r"\b(only|just)\s+(two|2|three|3)\b", text, re.I)
        if m:
            cap = 2 if "two" in m.group(0).lower() or "2" in m.group(0) else 3
            max_n = min(max_n, cap)
            min_n = min(min_n, cap)
    if state.get("exclude_personality"):
        max_n = max(min_n, max_n - 1)
    if state.get("reduce_coding"):
        max_n = max(min_n, max_n - 2)
    if state.get("conversation_goal") == "refine" or _is_refinement(messages, state):
        max_n = min(max_n, 8)

    min_n = max(1, min(min_n, max_n))
    max_n = min(10, max(1, max_n))
    return min_n, max_n


def user_confirmed_done(messages: list[dict], state: dict) -> bool:
    if not USER_DONE_PATTERN.search(_last(messages)):
        return False
    return bool(state.get("has_recommended"))


def next_clarification_for_slot(slot: str, archetype: str, state: dict, messages: list[dict]) -> str:
    """One combined question targeting the highest-priority missing slot."""
    prompts = {
        "audience": (
            "Who is this meant for — e.g., CXOs and directors, frontline managers, "
            "or a mixed leadership pool?"
        ),
        "hiring_purpose": (
            "Is this for selection against a benchmark, development for leaders already in role, "
            "or succession planning — and which leadership competencies matter most?"
        ),
        "operating_context": (
            "What is the role context — entry-level volume screening, experienced hires, "
            "and is the work mainly inbound calls or blended customer service?"
        ),
        "spoken_language": (
            "Before I shape the stack — what language are the calls in? "
            "That drives which spoken-language screen we use."
        ),
        "english_variant": (
            "SVAR has four English variants in the catalog: US, UK, Australian, and Indian accent. "
            "The choice matters because the screen is calibrated for the accent your callers will hear. "
            "Which fits your operation?"
        ),
        "stack_focus": (
            "Is this backend-leaning (Java/Spring/SQL heavy), frontend-heavy, or a balanced full-stack role?"
        ),
        "role_model": (
            "Is seniority closer to a senior IC (owns their service/design) or a tech lead "
            "(sets architecture across teams)?"
        ),
        "language_strategy": (
            "Knowledge tests are English-only while some personality measures support Spanish — "
            "will you run a hybrid (English knowledge + Spanish personality) or personality-only in Spanish?"
        ),
        "user_go_ahead": (
            "I can build a grounded shortlist from the closest catalog fits — "
            "shall I go ahead and list them?"
        ),
        "role_specificity": (
            "What is the specific role and seniority, and which 2–3 skills or competencies "
            "should the stack measure?"
        ),
        "competency_or_purpose": (
            "What should we prioritize — technical ability, communication, cognitive reasoning, "
            "personality fit, safety/dependability, or selection vs development purpose?"
        ),
    }
    missing = missing_slots(archetype, messages, state)
    if not missing:
        return prompts.get("role_specificity", "What role and competencies should we optimize for?")

    primary = missing[0]
    secondary = missing[1] if len(missing) > 1 else None
    q1 = prompts.get(primary, prompts["role_specificity"])
    if secondary and archetype in ("leadership_executive", "technical_jd"):
        q2 = prompts.get(secondary, "")
        if q2:
            return f"{q1} Also: {q2.lower()}"
    return q1


def generate_policy_clarification(state: dict, messages: list[dict]) -> str:
    from app.intent import extract_asked_slots, extract_answered_slots

    archetype = infer_archetype(messages, state)
    phase = conversation_phase(messages, state)
    asked = extract_asked_slots(messages)
    answered = extract_answered_slots(messages, state)

    if phase == "proposing":
        return next_clarification_for_slot("user_go_ahead", archetype, state, messages)

    if archetype == "leadership_executive" and _turns(messages) == 1:
        return "Happy to help narrow that down. Who is this meant for — CXOs and directors, frontline managers, or a mixed leadership pool?"

    missing = missing_slots(archetype, messages, state)
    for slot in missing:
        if slot in answered or slot in asked:
            continue
        return next_clarification_for_slot(slot, archetype, state, messages)

    domain_questions = {
        "contact_center": (
            "What language should spoken-communication assessments target, "
            "and is the work mainly inbound calls or blended customer service?"
        ),
        "technical_jd": (
            "Is this backend-leaning (Java/Spring/SQL), frontend-heavy, or balanced full-stack — "
            "and is seniority closer to senior IC or tech lead?"
        ),
        "leadership_executive": (
            "Is this for selection against a benchmark, development for leaders in role, "
            "or succession planning?"
        ),
        "sales_audit": (
            "Are you re-skilling individual contributors, frontline managers, or a mixed sales population?"
        ),
        "general_broad": (
            "What is the specific role and seniority, and which 2–3 competencies should the stack measure?"
        ),
    }
    if archetype in domain_questions and _turns(messages) <= 3:
        q = domain_questions[archetype]
        if q not in " ".join(m.get("content", "") for m in messages if m.get("role") == "assistant"):
            return q

    return next_clarification_for_slot(
        missing[0] if missing else "role_specificity",
        archetype,
        state,
        messages,
    )
