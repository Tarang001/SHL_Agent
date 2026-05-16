"""
Explicit per-turn intent classification — runs before retrieval / LLM.
"""
from __future__ import annotations

import re
from typing import Optional

# Intent labels (state["conversation_intent"])
VAGUE_INITIAL_QUERY = "VAGUE_INITIAL_QUERY"
CLARIFICATION_RESPONSE = "CLARIFICATION_RESPONSE"
SUFFICIENT_FOR_RECOMMENDATION = "SUFFICIENT_FOR_RECOMMENDATION"
REFINEMENT_REQUEST = "REFINEMENT_REQUEST"
COMPARISON_REQUEST = "COMPARISON_REQUEST"
FOLLOWUP_REASONING = "FOLLOWUP_REASONING"
OUT_OF_SCOPE = "OUT_OF_SCOPE"
PROMPT_INJECTION = "PROMPT_INJECTION"

MAX_TURNS = 8
FORCE_RECOMMEND_TURN = 7

EXPLICIT_COMPARISON = re.compile(
    r"\b("
    r"difference between|different from|what.?s the difference|how (do|does) .+ differ|"
    r"compare .+ (?:vs\.?|versus|and)|.+ (?:vs\.?|versus) .+|"
    r"which is better (?:for|between)|better (?:for|than)|is .+ different"
    r")\b",
    re.I,
)

REFINEMENT = re.compile(
    r"\b("
    r"add|drop|remove|replace|instead|update|change|without|also include|"
    r"actually|need less|need more|include|exclude|locking it in|"
    r"more leadership|shorter assessments?|include personality|"
    r"communication screening|reduce coding"
    r")\b",
    re.I,
)

# Single-token / short slot-fill answers — never comparisons
CLARIFICATION_FILL = re.compile(
    r"^(?:"
    r"us|usa|u\.s\.|uk|u\.k\.|australian|indian accent|"
    r"english|spanish|french|german|portuguese|bilingual|"
    r"backend|frontend|full[\s-]?stack|balanced|"
    r"inbound|outbound|blended|voice|non[\s-]?voice|"
    r"selection|development|succession|benchmark|hybrid|personality[\s-]?only|"
    r"yes|no|go ahead|confirmed|that works|"
    r"senior ic|tech lead|individual contributor"
    r")\.?$",
    re.I,
)

VAGUE_ONLY_PATTERNS = [
    re.compile(r"^we need a solution for senior leadership\.?$", re.I),
    re.compile(r"^hiring software engineers?\.?$", re.I),
    re.compile(r"^need assessments? for sales\.?$", re.I),
    re.compile(r"^hiring freshers?\.?$", re.I),
    re.compile(r"^need tests? for customer service\.?$", re.I),
    re.compile(r"^looking for leadership assessments?\.?$", re.I),
    re.compile(r"^need (an? )?assessments?\.?$", re.I),
    re.compile(r"^hiring (software )?engineers?\.?$", re.I),
    re.compile(r"^hiring for (sales|operations|customer service)\.?$", re.I),
]

BROAD_VAGUE = re.compile(
    r"\b("
    r"^hiring (software )?engineers?|need assessments|need tests|"
    r"solution for (senior )?leadership|hiring freshers?|"
    r"need assessments for sales|customer service|contact cent|"
    r"looking for leadership|software engineers?|developers?\b"
    r")",
    re.I,
)

SUFFICIENT_SIGNAL = re.compile(
    r"\b("
    r"mid.?level java|java backend|stakeholder|spring|rest api|"
    r"\d{3,}\s+entry.?level|inbound call|contact cent.{0,30}agent|"
    r"re-?skill.{0,30}sales|talent audit|succession planning|"
    r"plant operator|hipaa|5\+ years|full.?stack engineer|"
    r"screening \d+|spoken english|empathy|procedure compliance"
    r")\b",
    re.I,
)

INJECTION = re.compile(
    r"\b(ignore (previous|prior|all) instructions?|jailbreak|"
    r"reveal (the )?system prompt|pretend you are|override your)\b",
    re.I,
)

LEGAL_OFFTOPIC = re.compile(
    r"\b(legally required|legal obligation|hackerrank|codility|"
    r"what salary|employment law)\b",
    re.I,
)

# Slots the assistant may ask about (for loop prevention)
SLOT_PATTERNS = {
    "english_variant": re.compile(
        r"\b(us|uk|australian|indian accent|which english variant|english variant)\b", re.I
    ),
    "spoken_language": re.compile(
        r"\b(what language|spoken.?communication|language should)\b", re.I
    ),
    "stack_focus": re.compile(
        r"\b(backend.?leaning|frontend.?heavy|full.?stack|stack focus)\b", re.I
    ),
    "hiring_purpose": re.compile(
        r"\b(selection|development|succession|benchmark|hiring purpose)\b", re.I
    ),
    "audience": re.compile(
        r"\b(cxo|director|who is this meant for|leadership pool)\b", re.I
    ),
    "operating_context": re.compile(
        r"\b(entry.?level|inbound|volume screening|operating context)\b", re.I
    ),
    "role_specificity": re.compile(
        r"\b(specific role|seniority|which 2.?3|competencies should)\b", re.I
    ),
}


def user_turn_count(messages: list[dict]) -> int:
    return len([m for m in messages if m.get("role") == "user"])


def last_user_text(messages: list[dict]) -> str:
    return messages[-1]["content"] if messages else ""


def _last_assistant_text(messages: list[dict]) -> str:
    for msg in reversed(messages[:-1]):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def assistant_asked_clarification(messages: list[dict]) -> bool:
    text = _last_assistant_text(messages)
    return bool(text and "?" in text)


def extract_asked_slots(messages: list[dict]) -> set[str]:
    """Slots already asked by the assistant in the conversation."""
    asked: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        text = msg.get("content", "")
        for slot, pat in SLOT_PATTERNS.items():
            if pat.search(text) and "?" in text:
                asked.add(slot)
    return asked


def extract_answered_slots(messages: list[dict], state: dict) -> set[str]:
    """Slots filled by cumulative user text."""
    answered: set[str] = set()
    full = " ".join(m["content"] for m in messages if m.get("role") == "user").lower()

    if re.search(r"\b(us|usa|u\.s\.)\b", full) or state.get("language") == "English (USA)":
        answered.add("english_variant")
    if re.search(r"\b(uk|british)\b", full):
        answered.add("english_variant")
    if re.search(r"\benglish\b", full):
        answered.add("spoken_language")
    if re.search(r"\b(inbound|outbound|500|contact cent|call cent)\b", full, re.I):
        answered.add("operating_context")
    if re.search(r"\b(java|spring|python|backend|frontend|full.?stack)\b", full, re.I):
        answered.add("stack_focus")
        answered.add("role_specificity")
    if re.search(r"\b(selection|development|succession|benchmark|audit|re-?skill)\b", full, re.I):
        answered.add("hiring_purpose")
    if re.search(r"\b(cxo|director|executive|15\+)\b", full, re.I):
        answered.add("audience")
    if state.get("seniority") or SUFFICIENT_SIGNAL.search(full):
        answered.add("role_specificity")

    return answered


def is_clarification_fill(message: str) -> bool:
    text = message.strip()
    if CLARIFICATION_FILL.match(text):
        return True
    if len(text.split()) <= 3 and not EXPLICIT_COMPARISON.search(text):
        if re.match(
            r"^(english|spanish|us|uk|backend|frontend|inbound|outbound|selection|development)$",
            text,
            re.I,
        ):
            return True
    return False


def is_explicit_comparison(message: str, *, has_prior: bool = False) -> bool:
    """Comparison only when user explicitly contrasts two things — not slot fills."""
    text = message.strip()
    if is_clarification_fill(text):
        return False
    if not EXPLICIT_COMPARISON.search(text):
        return False
    if re.match(
        r"^(english|us|uk|backend|frontend|freshers?|leadership|sales|developers?)$",
        text,
        re.I,
    ):
        return False
    from app.followup import _extract_comparison_sides

    if _extract_comparison_sides(text):
        return True
    if has_prior and re.search(r"\b(opq|verify|gsa|dsi|svar)\b", text, re.I):
        if re.search(r"\b(and|vs\.?|versus|difference|compare|better)\b", text, re.I):
            return True
    return False


def is_vague_initial(messages: list[dict], state: dict) -> bool:
    if user_turn_count(messages) > 1:
        return False
    last = last_user_text(messages).strip()
    for pat in VAGUE_ONLY_PATTERNS:
        if pat.search(last):
            return True
    if len(last.split()) <= 12 and not SUFFICIENT_SIGNAL.search(last):
        if BROAD_VAGUE.search(last) or state.get("conversation_archetype") == "general_broad":
            if not SUFFICIENT_SIGNAL.search(last):
                return True
    return False


def has_sufficient_context(messages: list[dict], state: dict) -> bool:
    """True only when archetype-required slots are filled — not on volume/role hints alone."""
    from app.conversation_policy import IMMEDIATE_ARCHETYPES, infer_archetype, missing_slots

    archetype = state.get("conversation_archetype") or infer_archetype(messages, state)
    missing = missing_slots(archetype, messages, state)

    if archetype in IMMEDIATE_ARCHETYPES:
        return True

    # Archetypes with mandatory slots: all must be filled before recommending
    if missing:
        return False

    if archetype == "contact_center":
        answered = extract_answered_slots(messages, state)
        return "english_variant" in answered or state.get("language") == "English (USA)"

    if archetype == "leadership_executive":
        answered = extract_answered_slots(messages, state)
        return "audience" in answered and "hiring_purpose" in answered

    full = " ".join(m["content"] for m in messages if m.get("role") == "user")
    if SUFFICIENT_SIGNAL.search(full) and archetype not in (
        "contact_center",
        "leadership_executive",
        "general_broad",
    ):
        return True
    if state.get("strong_signal_count", 0) >= 2:
        from app.state import has_sufficient_hiring_signal

        return has_sufficient_hiring_signal(state, messages)
    return not missing


def classify_intent(messages: list[dict], state: dict) -> str:
    """
    Classify the latest user turn into exactly one intent label.
    Must run after state reconstruction (archetype, tags, memory attached).
    """
    last = last_user_text(messages)
    turns = user_turn_count(messages)
    has_prior = bool(
        state.get("has_recommendations")
        or state.get("has_recommended")
        or state.get("recommended_tests")
    )

    if INJECTION.search(last):
        return PROMPT_INJECTION
    if LEGAL_OFFTOPIC.search(last):
        return OUT_OF_SCOPE

    if turns >= FORCE_RECOMMEND_TURN and not has_prior:
        if is_explicit_comparison(last, has_prior=has_prior):
            return COMPARISON_REQUEST if has_prior else SUFFICIENT_FOR_RECOMMENDATION
        return SUFFICIENT_FOR_RECOMMENDATION

    from app.followup import shortlist_likely_issued

    has_shortlist = has_prior or shortlist_likely_issued(messages, state)

    if has_shortlist:
        mode = state.get("post_shortlist_mode", "none")
        if mode == "refine":
            return REFINEMENT_REQUEST
        if mode == "informational":
            return COMPARISON_REQUEST
        if REFINEMENT.search(last):
            return REFINEMENT_REQUEST
        if is_explicit_comparison(last, has_prior=True):
            return COMPARISON_REQUEST
        if is_followup_reasoning(last, state):
            return FOLLOWUP_REASONING
        if re.search(r"\b(different|compare|versus|vs\.?|better|should we use|both)\b", last, re.I):
            return FOLLOWUP_REASONING

    if assistant_asked_clarification(messages) and is_clarification_fill(last):
        return CLARIFICATION_RESPONSE

    if assistant_asked_clarification(messages) and len(last.split()) <= 6:
        if not is_explicit_comparison(last, has_prior=has_prior):
            if not REFINEMENT.search(last):
                return CLARIFICATION_RESPONSE

    if is_explicit_comparison(last, has_prior=has_prior) and has_shortlist:
        return COMPARISON_REQUEST

    if REFINEMENT.search(last) and has_prior:
        return REFINEMENT_REQUEST

    if has_sufficient_context(messages, state):
        return SUFFICIENT_FOR_RECOMMENDATION

    if is_vague_initial(messages, state) or turns == 1:
        from app.conversation_policy import should_recommend_slots

        if not should_recommend_slots(messages, state) and not has_sufficient_context(messages, state):
            return VAGUE_INITIAL_QUERY

    from app.conversation_policy import should_recommend_slots

    if should_recommend_slots(messages, state):
        return SUFFICIENT_FOR_RECOMMENDATION

    return VAGUE_INITIAL_QUERY


def is_followup_reasoning(message: str, state: dict) -> bool:
    if not state.get("has_recommendations") and not state.get("recommended_tests"):
        return False
    if is_explicit_comparison(message, has_prior=True):
        return False
    if REFINEMENT.search(message):
        return False
    patterns = re.compile(
        r"\b(should (we|i)|can (this|it)|is this|worth|suitab|when to use|"
        r"why both|do we need|can i use|for freshers?|personality or)\b",
        re.I,
    )
    return bool(patterns.search(message))


def intent_to_policy_action(intent: str) -> str:
    return {
        VAGUE_INITIAL_QUERY: "clarify",
        CLARIFICATION_RESPONSE: "clarify_or_recommend",
        SUFFICIENT_FOR_RECOMMENDATION: "recommend",
        REFINEMENT_REQUEST: "refine",
        COMPARISON_REQUEST: "compare",
        FOLLOWUP_REASONING: "discuss",
        OUT_OF_SCOPE: "refuse",
        PROMPT_INJECTION: "refuse",
    }.get(intent, "clarify")


def should_end_conversation(messages: list[dict], state: dict, result: dict) -> bool:
    """Turn 8+ force close; after answered follow-up with prior shortlist."""
    turns = user_turn_count(messages)
    if turns >= MAX_TURNS:
        return True
    if user_confirmed_done(messages, state):
        return True
    has_prior = bool(state.get("recommended_tests") or state.get("has_recommendations"))
    intent = state.get("conversation_intent", "")
    if has_prior and intent == CLARIFICATION_RESPONSE and result.get("recommendations"):
        return True
    if has_prior and intent == SUFFICIENT_FOR_RECOMMENDATION and result.get("recommendations"):
        return turns >= 2
    return False


def user_confirmed_done(messages: list[dict], state: dict) -> bool:
    from app.conversation_policy import USER_DONE_PATTERN

    if not USER_DONE_PATTERN.search(last_user_text(messages)):
        return False
    return bool(state.get("has_recommended") or state.get("has_recommendations"))