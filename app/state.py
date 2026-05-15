"""
Conversation state extraction — deterministic parsing layer.
Extracts structured constraints from conversation history without LLM.
"""
import re
from typing import Optional

from app.conversation_policy import (
    conversation_phase,
    generate_policy_clarification,
    infer_archetype,
    missing_slots,
    should_clarify_slots,
    should_recommend_slots,
    user_confirmed_done,
)

RECOMMEND_THRESHOLD = 5
MIN_STRONG_SIGNALS = 2
AMBIGUITY_CLARIFY_THRESHOLD = 3

# Broad role families — not actionable without specialization + another signal
BROAD_ROLE_TAXONOMY = re.compile(
    r'\b('
    r'leadership|leaders?|senior leadership|leadership (assessments?|solutions?|roles?|hiring)|'
    r'solution for .{0,25}leadership|'
    r'management|managers?|executives?|'
    r'software engineers?|developers?|engineers?|programmers?|coders?|'
    r'freshers?|graduates?|graduate hires?|'
    r'customer support|customer service|call cent(er|re)|contact cent(er|re)|'
    r'operations|sales hires?|sales hiring|marketing|human resources|\bhr\b|finance|'
    r'analysts?|support (staff|roles?|hires?)|'
    r'assessments? for managers?|need tests? for managers?|hiring for operations|'
    r'we need sales|need assessments\.?|hiring software engineers?'
    r')\b',
    re.I,
)

# Narrowing patterns — actionable role/function specificity
ACTIONABLE_ROLE_PATTERN = re.compile(
    r'\b('
    r'java backend|backend engineer|frontend engineer|full.?stack developer|'
    r'devops engineer|platform engineer|inside sales|field sales|'
    r'plant operator|chemical (facility|manufacturing)|warehouse supervisor|'
    r'call cent(er|re) agents?|contact cent(er|re) agents?|'
    r'call cent(er|re) (agent|associate|rep)|contact cent(er|re) (agent|associate)|'
    r'customer support rep|admin assistant|data analyst|business analyst|'
    r'graduate management trainee|management trainee scheme|'
    r'\bcxo\b|\bc-suite\b|director.level|15\+ years|'
    r'rust engineer|high.performance networking|networking infrastructure|'
    r'bilingual healthcare admin|hipaa compliance|'
    r're-?skill.{0,30}sales|talent audit'
    r')\b',
    re.I,
)

COMPETENCY_SIGNAL_PATTERN = re.compile(
    r'\b('
    r'stakeholder|analytical reasoning|problem solving|critical thinking|'
    r'spring boot|microservices|empathy|negotiation|persuasion|'
    r'coding ability|spoken english|procedure compliance|safety compliance|'
    r'safety.critical|dependability|communication skills|customer.?facing|'
    r'leadership potential|management potential|strategic thinking|people management|'
    r'personality fit|cognitive (ability|reasoning|test)|situational judgement|'
    r'selection against|leadership benchmark|assessment battery|inbound calls?|'
    r'high.performance|networking infrastructure'
    r')\b',
    re.I,
)

DOMAIN_SIGNAL_PATTERN = re.compile(
    r'\b('
    r'healthcare|banking|financial services|manufacturing|chemical|saas|telecom|'
    r'retail|insurance|pharma|logistics|oil and gas|energy sector|'
    r'south texas|patient records|hipaa|inbound call'
    r')\b',
    re.I,
)

PURPOSE_SIGNAL_PATTERN = re.compile(
    r'\b('
    r'screening|succession planning|promotion|leadership development|'
    r'campus hiring|selection against|benchmark|talent audit|re-?skill|'
    r'development program|hiring for selection|comparing candidates|'
    r'assessment battery|trainee scheme'
    r')\b',
    re.I,
)

# ── Keyword patterns ─────────────────────────────────────────────────────────

SENIORITY_PATTERNS = [
    (r'\b(entry.?level|junior|fresh(er)?|0[-–]\d year)\b', 'entry'),
    (r'\b(mid.?level|intermediate|3[-–]5 year|4[-–]6 year|around [23456] year)\b', 'mid'),
    (r'\b(senior|sr\.?|5\+|7\+|8\+|experienced)\b', 'senior'),
    (r'\b(lead|principal|staff)\b', 'lead'),
    (r'\b(manager|management)\b', 'manager'),
    (r'\b(director)\b', 'director'),
    (r'\b(executive|cxo|ceo|cto|cfo|coo|vp |vice president)\b', 'executive'),
    (r'\b(graduate|grad scheme|fresh grad|recent grad|management trainee)\b', 'graduate'),
]

SPECIFIC_TECH_PATTERN = re.compile(
    r'\b('
    r'java|python|rust|spring|sql|react|node\.?js|angular|aws|docker|kubernetes|'
    r'excel|word|hipaa|\.net|c\+\+|c#|linux|networking|full.?stack|backend|frontend|'
    r'g\+|verify interactive|opq|svar|bilingual'
    r')\b',
    re.I,
)

GENERIC_ROLE_PATTERN = re.compile(
    r'\b('
    r'software engineers?|developers?|engineers?|programmers?|coders?|'
    r'tech roles?|tech hiring|coding tests?|software tests?|developer assessments?'
    r')\b',
    re.I,
)

# Role labels that still lack competency / hiring-context signal on their own
VAGUE_ROLE_LABEL_PATTERN = re.compile(
    r'\b('
    r'software engineers?|backend engineers?|frontend engineers?|full.?stack engineers?|'
    r'freshers?|graduate hires?|graduates?|engineers?|developers?|'
    r'customer support|call cent(er|re) agents?|marketing roles?|'
    r'sales executives?|analysts?|operations managers?|operations hires?|'
    r'sales hires?|warehouse supervisors?|data analysts?'
    r')\b',
    re.I,
)

# Roles / intents specific enough to recommend without extra competency detail
SUFFICIENT_ROLE_ALONE_PATTERN = re.compile(
    r'\b('
    r'graduate management trainee|management trainee scheme|'
    r'plant operator|chemical facilit|contact cent(er|re) agent.*inbound|'
    r're-?skill.{0,30}sales|sales organization.{0,30}audit|'
    r'bilingual healthcare admin|hipaa'
    r')\b',
    re.I,
)

DEPARTMENT_ONLY_PATTERN = re.compile(
    r'^(hiring for |need )?(sales|operations|marketing|engineering|it|hr|finance)\s*(hires?|roles?|team)?\.?$',
    re.I,
)

SENIORITY_ONLY_PATTERN = re.compile(
    r'^(hiring |need )?(freshers?|graduates?|entry.?level hires?|senior hires?|experienced hires?)\.?$',
    re.I,
)

BROAD_VAGUE_PATTERNS = [
    re.compile(r"^i need (an? )?assessments?\.?$", re.I),
    re.compile(r"^need (an? )?(hiring )?tests?\.?$", re.I),
    re.compile(r"^help me find (an? )?assessments?\.?$", re.I),
    re.compile(r"^what assessments? do you have\??$", re.I),
    re.compile(r"^recommend something\.?$", re.I),
    re.compile(r"\bneed (an? )?assessments?\b", re.I),
    re.compile(r"\bneed (hiring )?tests?\b", re.I),
    re.compile(r"\bassessments? for (software )?engineers?\b", re.I),
    re.compile(r"\bfor developers?\b", re.I),
    re.compile(r"\bneed (coding|software) tests?\b", re.I),
    re.compile(r"\bhiring (for )?(tech|software) roles?\b", re.I),
    re.compile(r"\btests? for developers?\b", re.I),
    re.compile(r"^hiring engineers?\.?$", re.I),
]

SPECIFIC_ROLE_PATTERNS = [
    (r'\b(java|python|rust|spring|sql|react|node)\s*(/|\s+and\s+|\s+)?(developer|engineer)', 'technical hire'),
    (r'\b(full.?stack|backend|frontend)\s+(developer|engineer)', 'technical hire'),
    (r'\b(contact cent(re|er)|call cent(re|er))\s+(agent|rep)', 'contact center'),
    (r'\b(customer service|customer support)\s+(agent|rep)', 'customer service'),
    (r'\b(plant operator|chemical|manufacturing)\b', 'operations'),
    (r'\b(admin(istrative)? assistant|office admin)\b', 'administrative'),
    (r'\b(healthcare|medical admin|patient records)\b', 'healthcare'),
    (r'\b(sales|account executive|business development)\b', 'sales'),
    (r'\b(data analysts?|business analysts?)\b', 'analytics'),
    (r'\b(graduate|management trainee)\b', 'graduate scheme'),
    (r'\b(cxo|ceo|cto|cfo|director|vp)\b', 'leadership'),
    (r'\b(nurse|clinical)\b', 'healthcare'),
    (r'\b(manager|managers|supervisor)\b', 'management'),
]

ASSESSMENT_TYPE_INTENT = re.compile(
    r'\b(cognitive tests?|personality tests?|simulation tests?|'
    r'knowledge tests?|ability tests?|spoken english|sjt|reasoning tests?)\b',
    re.I,
)

HIGH_SIGNAL_DOMAIN_TAGS = frozenset({
    'safety', 'contact_center', 'customer_service', 'healthcare', 'hipaa',
    'graduate', 'sales', 'executive', 'leadership', 'admin', 'microsoft_office',
    'reskilling',
})

SPECIFIC_TECH_TAGS = frozenset({
    'java', 'python', 'rust', 'spring', 'sql', 'aws', 'docker', 'angular',
    'hipaa', 'microsoft_office', 'networking', 'spoken_language',
})

EXCLUDE_PATTERNS = [
    (r'\b(no personality|drop (the )?personality|remove (the )?opq|skip personality|without personality)\b', 'personality'),
    (r'\b(no cognitive|drop (the )?cognitive|remove.*verify|skip cognitive)\b', 'cognitive'),
    (r'\b(no simulation|drop simulation|skip simulation)\b', 'simulation'),
    (r'\b(remove|drop|exclude|without)\b.{0,40}\b(coding|code|technical)\b', 'coding'),
    (r'\b(less senior|more junior|lower level)\b', 'seniority_down'),
    (r'\b(more senior|higher level)\b', 'seniority_up'),
]

INCLUDE_PATTERNS = [
    (r'\b(add personality|include personality|with personality|personality test)\b', 'personality'),
    (r'\b(add (a )?cognitive|include cognitive|cognitive test|reasoning test)\b', 'cognitive'),
    (r'\b(add simulation|include simulation|with simulation)\b', 'simulation'),
    (r'\b(add|include|also need|with)\b.{0,40}\b(communication|spoken|language)\b', 'communication'),
    (r'\b(add|include)\b.{0,40}\b(coding|technical|programming)\b', 'coding'),
]

PROMPT_INJECTION_PATTERNS = [
    r'\bignore (previous|prior|all) instructions?\b',
    r'\bforget (your|the) (instruction|prompt|rules)s?\b',
    r'\b(you are now|act as) (a |an )?(general|helpful|unrestricted)\b',
    r'\b(reveal|show|print|dump)\b.{0,30}\b(system prompt|hidden prompt|instructions?)\b',
    r'\bjailbreak\b',
    r'\bpretend you are\b',
    r'\boverride (your|the) (role|instructions?)\b',
]

OFFTOPIC_PATTERNS = [
    (r'\b(legally required|does this (test |assessment )?satisfy|legal obligation|regulatory requirement|required under hipaa)\b', 'legal'),
    (r'\b(what salary|how much (should|to) pay|compensation package|pay range)\b', 'salary'),
    (r'\b(how to interview|interview questions? for|hiring policy|employment law)\b', 'hr_advice'),
    (r'\b(best ats|applicant tracking|workday|greenhouse|lever)\b', 'ats'),
    (r'\b(hackerrank|leetcode|codility|use (another|different) (tool|platform|service))\b', 'competitor'),
    (r'\b(general hiring advice|recruitment strategy)\b', 'hr_advice'),
]

REFINEMENT_PATTERNS = re.compile(
    r'\b(add|drop|remove|replace|instead|update|change|without|also include|keep|'
    r'actually|need less|need more|include|exclude)\b',
    re.I,
)

COMPARISON_PATTERNS = re.compile(
    r'\b(difference between|compare|comparison|vs\.?|versus|which is better|'
    r'how (do|does) .+ differ|what.?s the difference)\b',
    re.I,
)

TAG_QUERY_EXPANSIONS = {
    "executive": "OPQ32r leadership personality executive selection",
    "leadership": "OPQ leadership competency executive",
    "safety": "dependability safety DSI procedure compliance manufacturing",
    "contact_center": "contact center customer service SVAR call simulation entry level",
    "customer_service": "customer service phone simulation contact center",
    "graduate": "graduate scenarios Verify Interactive G+ cognitive personality SJT",
    "sales": "global skills assessment sales transformation OPQ MQ",
    "coding": "Smart Interview Live Coding programming technical",
    "rust": "Rust Smart Interview Live Coding Linux networking infrastructure",
    "java": "Core Java Spring RESTful Web Services SQL",
    "spring": "Spring Core Java REST API",
    "sql": "SQL database",
    "networking": "networking implementation Linux programming",
    "microsoft_office": "Microsoft Excel Word MS Office admin",
    "admin": "admin assistant Excel Word Microsoft Office",
    "healthcare": "HIPAA medical terminology healthcare patient records Word",
    "hipaa": "HIPAA security compliance healthcare",
    "spoken_language": "SVAR spoken English language assessment",
    "reskilling": "global skills development sales transformation talent audit",
}


def user_turn_count(messages: list[dict]) -> int:
    return len([m for m in messages if m["role"] == "user"])


def user_text(messages: list[dict]) -> str:
    return " ".join(m["content"] for m in messages if m["role"] == "user").lower()


def last_user_text(messages: list[dict]) -> str:
    return messages[-1]["content"].lower()


def is_vague(text: str) -> bool:
    """Legacy helper — true for ultra-broad single-line intents."""
    t = text.strip().lower()
    for pattern in BROAD_VAGUE_PATTERNS:
        if pattern.search(t):
            return True
    if len(t.split()) <= 4 and not SPECIFIC_TECH_PATTERN.search(t):
        return True
    return False


def is_prompt_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in PROMPT_INJECTION_PATTERNS)


def is_offtopic(text: str) -> Optional[str]:
    t = text.lower()
    for pattern, category in OFFTOPIC_PATTERNS:
        if re.search(pattern, t):
            return category
    if is_prompt_injection(t):
        return "injection"
    return None


def is_refusal_needed(text: str) -> Optional[str]:
    """Map off-topic categories to refusal response types."""
    category = is_offtopic(text)
    if not category:
        return None
    if category == "legal":
        return "legal"
    if category == "injection":
        return "injection"
    return "out_of_scope"


def is_comparison_query(state: dict, messages: list[dict]) -> bool:
    if state.get("conversation_goal") == "compare":
        return True
    return bool(COMPARISON_PATTERNS.search(user_text(messages)))


def is_refinement(state: dict, messages: list[dict]) -> bool:
    if state.get("conversation_goal") == "refine":
        return True
    if not state.get("has_recommended"):
        return False
    return bool(REFINEMENT_PATTERNS.search(last_user_text(messages)))


def _has_specific_role(user_text_lower: str) -> bool:
    if GENERIC_ROLE_PATTERN.search(user_text_lower):
        if not SPECIFIC_TECH_PATTERN.search(user_text_lower):
            for pattern, _ in SPECIFIC_ROLE_PATTERNS:
                if re.search(pattern, user_text_lower, re.I):
                    return True
            return False
    return any(re.search(p, user_text_lower, re.I) for p, _ in SPECIFIC_ROLE_PATTERNS)


def _has_specific_tech_stack(state: dict, user_text_lower: str) -> bool:
    tags = set(state.get("inferred_tags", []))
    specific = tags.intersection(SPECIFIC_TECH_TAGS)
    if specific:
        return True
    if SPECIFIC_TECH_PATTERN.search(user_text_lower):
        if tags == {"coding"} and GENERIC_ROLE_PATTERN.search(user_text_lower):
            return bool(state.get("seniority"))
        return True
    return False


def _has_behavioral_signal(user_text_lower: str, state: dict) -> bool:
    if re.search(
        r'\b(communication|stakeholder|empathy|personality|cognitive|simulation|'
        r'customer service|customer.?facing|safety|dependability|spoken english|'
        r'bilingual|leadership|negotiation|persuasion|analytical reasoning|'
        r'problem solving|judgment|judgement|interpersonal|service orientation)\b',
        user_text_lower,
    ):
        return True
    return bool(
        state.get("include_personality")
        or state.get("include_cognitive")
        or state.get("include_simulation")
    )


def _has_hiring_context(user_text_lower: str) -> bool:
    return bool(re.search(
        r'\b(selection|development|screening|benchmark|hiring for|'
        r'trainee scheme|talent audit|re-?skill|assessment battery|'
        r'inbound calls?|safety.critical|top priority|day.one)\b',
        user_text_lower,
    ))


def _has_competency_or_goal_signal(user_text_lower: str, state: dict) -> bool:
    if COMPETENCY_SIGNAL_PATTERN.search(user_text_lower):
        return True
    if ASSESSMENT_TYPE_INTENT.search(user_text_lower):
        return True
    if _has_behavioral_signal(user_text_lower, state):
        if is_broad_role_category(user_text_lower, {}):
            return bool(COMPETENCY_SIGNAL_PATTERN.search(user_text_lower))
        return True
    tags = set(state.get("inferred_tags", []))
    if tags.intersection({"safety", "hipaa", "reskilling"}) and re.search(
        r'\b(safety|hipaa|dependability|procedure|audit|re-?skill)\b', user_text_lower, re.I
    ):
        return True
    return False


def is_broad_role_category(text: str, state: dict) -> bool:
    """True when the user named a broad job family, not an actionable role."""
    if re.search(
        r'\b(java|python|rust|spring|sql|react|angular|node\.?js|aws|docker|kubernetes|'
        r'golang|scala|c\+\+)\s+(backend\s+)?engineers?\b',
        text,
        re.I,
    ):
        return False
    if _has_specific_tech_stack(state, text) and re.search(r'\bengineers?\b', text, re.I):
        return False
    if BROAD_ROLE_TAXONOMY.search(text):
        if ACTIONABLE_ROLE_PATTERN.search(text):
            return False
        return True
    if state.get("role") in (
        "leadership", "management", "generic technical", "sales", "graduate scheme", "analytics"
    ):
        if not ACTIONABLE_ROLE_PATTERN.search(text) and not _has_specific_tech_stack(state, text):
            return True
    if GENERIC_ROLE_PATTERN.search(text) and not ACTIONABLE_ROLE_PATTERN.search(text):
        if not _has_specific_tech_stack(state, text):
            return True
    return False


def _signal_role_specificity(text: str, state: dict) -> bool:
    if SUFFICIENT_ROLE_ALONE_PATTERN.search(text):
        return True
    if ACTIONABLE_ROLE_PATTERN.search(text):
        return True
    if _has_specific_tech_stack(state, text):
        return True
    for pattern, role in SPECIFIC_ROLE_PATTERNS:
        if role in ("management", "leadership", "sales"):
            continue
        if re.search(pattern, text, re.I):
            return True
    if re.search(
        r'\b(java|python|rust|spring|sql)\s*(/|\s+and\s+|\s+)?(developer|engineer)\b',
        text,
        re.I,
    ):
        return True
    return False


def _signal_competencies(text: str, state: dict) -> bool:
    return _has_competency_or_goal_signal(text, state)


def _signal_seniority(state: dict, text: str, broad_role: bool) -> bool:
    if broad_role:
        return False
    if state.get("seniority"):
        return True
    return bool(re.search(
        r'\b(entry.?level|mid.?level|senior ic|senior manager|experienced hire)\b',
        text,
        re.I,
    ))


def _signal_domain(state: dict, text: str) -> bool:
    if state.get("industry"):
        return True
    if DOMAIN_SIGNAL_PATTERN.search(text):
        return True
    tags = set(state.get("inferred_tags", []))
    return bool(tags.intersection({
        "healthcare", "hipaa", "safety", "contact_center", "customer_service", "reskilling",
    }) and re.search(
        r'\b(healthcare|patient|hipaa|safety|contact|call cent|inbound|audit|re-?skill)\b',
        text,
        re.I,
    ))


def _signal_purpose(text: str) -> bool:
    if PURPOSE_SIGNAL_PATTERN.search(text):
        return True
    return _has_hiring_context(text)


def count_strong_signals(state: dict, messages: list[dict]) -> int:
    """Count hiring signals (need >= MIN_STRONG_SIGNALS to recommend)."""
    text = user_text(messages)
    broad = is_broad_role_category(text, state)
    signals = [
        _signal_role_specificity(text, state),
        _signal_competencies(text, state),
        _signal_seniority(state, text, broad),
        _signal_domain(state, text),
        _signal_purpose(text),
    ]
    return sum(signals)


def ambiguity_score(state: dict, messages: list[dict]) -> int:
    """Higher => must clarify before recommending."""
    text = user_text(messages)
    strong = count_strong_signals(state, messages)
    if strong >= MIN_STRONG_SIGNALS and _signal_role_specificity(text, state):
        return 0
    score = 0
    if is_broad_role_category(text, state):
        score += 2
    if _matches_broad_vague_intent(messages[-1]["content"]):
        score += 2
    if not _signal_competencies(text, state):
        score += 1
    if not _signal_purpose(text):
        score += 1
    if is_broad_role_category(text, state) and not _signal_role_specificity(text, state):
        score += 1
    if strong < MIN_STRONG_SIGNALS:
        score += 1
    return score


def is_role_only_vague(messages: list[dict], state: dict) -> bool:
    """True when hiring intent is underspecified (broad role / <2 strong signals)."""
    if has_sufficient_hiring_signal(state, messages):
        return False
    return (
        is_broad_role_category(user_text(messages), state)
        or ambiguity_score(state, messages) >= AMBIGUITY_CLARIFY_THRESHOLD
        or count_strong_signals(state, messages) < MIN_STRONG_SIGNALS
    )


def has_sufficient_hiring_signal(state: dict, messages: list[dict]) -> bool:
    """
    Recommend only when >= MIN_STRONG_SIGNALS from:
    role specificity, competencies, seniority (non-broad), domain, purpose.
    """
    text = user_text(messages)
    last = last_user_text(messages)

    if len(text.split()) >= 25 or re.search(
        r'\b(responsibilities|requirements|qualifications|job description|jd:|'
        r"here.?s the jd|role profile|must have|nice to have|5\+ years)\b",
        text,
        re.I,
    ):
        return True

    strong = count_strong_signals(state, messages)
    if strong >= MIN_STRONG_SIGNALS:
        if is_broad_role_category(text, state):
            return _signal_role_specificity(text, state) and _signal_competencies(text, state)
        return True

    if re.search(
        r'\b(senior )?rust engineer.{0,80}(networking|infrastructure)\b',
        text,
        re.I,
    ):
        return True

    if re.search(
        r'\b(mid.?level|senior)\s+(java|python|rust)\s*(backend\s+)?engineer.{0,80}'
        r'(spring|stakeholder|backend|networking)\b',
        text,
        re.I,
    ):
        return True

    if re.search(r'\b(core java|spring|rest api).{0,80}(sql|docker|aws)\b', text, re.I):
        return True

    if re.search(
        r'\b(graduate|entry).{0,40}(call cent|contact cent).{0,40}(spoken english|empathy)\b',
        text,
        re.I,
    ):
        return True

    if re.search(
        r'\b(plant operator|warehouse supervisor).{0,50}(safety|chemical|dependability)\b',
        text,
        re.I,
    ):
        return True

    if re.search(
        r'\b(leadership|personality).{0,30}assessments?.{0,30}(senior )?sales managers?\b',
        text,
        re.I,
    ):
        return True

    if re.search(
        r'\b(customer support|contact cent).{0,60}'
        r'(spoken english|empathy|communication)\b',
        text,
        re.I,
    ):
        return True

    if re.search(
        r'\b(data analyst|business analyst).{0,40}(sql|analytical|excel)\b',
        text,
        re.I,
    ):
        return True

    if re.search(
        r'\bgraduate management trainee\b', text, re.I
    ) and re.search(r'\b(cognitive|personality|sjt|battery|leadership)\b', text, re.I):
        return True

    return False


def _matches_broad_vague_intent(text: str) -> bool:
    t = text.strip().lower()
    for pattern in BROAD_VAGUE_PATTERNS:
        if pattern.search(t):
            return True
    if GENERIC_ROLE_PATTERN.search(t) and not SPECIFIC_TECH_PATTERN.search(t):
        if not re.search(r'\b(entry|mid|senior|lead|manager|director|executive)\b', t):
            return True
    return False


def state_confidence_score(state: dict, messages: list[dict]) -> int:
    """Deterministic hiring-context confidence (higher => ready to recommend)."""
    text = user_text(messages)
    last = last_user_text(messages)
    strong = count_strong_signals(state, messages)

    if not has_sufficient_hiring_signal(state, messages):
        return min(strong, 2)

    score = 4 + strong

    if len(last.split()) >= 25:
        score += 2
    if _signal_role_specificity(text, state):
        score += 1
    if _has_specific_tech_stack(state, text):
        score += 1

    if is_broad_role_category(text, state):
        score = min(score, 3)

    if user_turn_count(messages) >= 2:
        score += 1

    return score


def should_recommend(state: dict, messages: list[dict]) -> bool:
    """True when scenario slots are filled or user approved a proposed stack."""
    if is_offtopic(last_user_text(messages)):
        return False
    if is_comparison_query(state, messages) and not state.get("has_recommended"):
        return False
    if user_turn_count(messages) >= 7 and should_recommend_slots(messages, state):
        return True
    return should_recommend_slots(messages, state)


def should_clarify(state: dict, messages: list[dict]) -> bool:
    """True when critical hiring slots are still unfilled."""
    return should_clarify_slots(messages, state)


def generate_clarification_question(state: dict, messages: list[dict]) -> str:
    return generate_policy_clarification(state, messages)


def extract_constraints(messages: list[dict]) -> dict:
    """Parse refinement constraints from the latest user turn."""
    last = last_user_text(messages)
    constraints = {
        "include": [],
        "exclude": [],
        "seniority_hint": None,
        "communication": False,
        "reduce_coding": False,
        "leadership": False,
    }

    for pattern, category in INCLUDE_PATTERNS:
        if re.search(pattern, last, re.I):
            constraints["include"].append(category)
            if category == "communication":
                constraints["communication"] = True

    for pattern, category in EXCLUDE_PATTERNS:
        if re.search(pattern, last, re.I):
            constraints["exclude"].append(category)
            if category == "seniority_down":
                constraints["seniority_hint"] = "entry"
            elif category == "seniority_up":
                constraints["seniority_hint"] = "senior"

    if re.search(r"\b(less coding|reduce coding|not coding|without coding|non.?technical focus)\b", last, re.I):
        constraints["reduce_coding"] = True
    if re.search(r"\b(leadership|managerial|management potential)\b", last, re.I):
        constraints["leadership"] = True

    return constraints


def merge_constraints(state: dict, constraints: dict) -> dict:
    """Apply extracted refinement constraints onto conversation state."""
    for category in constraints.get("include", []):
        if category == "personality":
            state["include_personality"] = True
            state["exclude_personality"] = False
        elif category == "cognitive":
            state["include_cognitive"] = True
        elif category == "simulation":
            state["include_simulation"] = True
        elif category == "coding":
            if "coding" not in state["inferred_tags"]:
                state["inferred_tags"].append("coding")
        elif category == "communication":
            if "spoken_language" not in state["inferred_tags"]:
                state["inferred_tags"].append("spoken_language")

    for category in constraints.get("exclude", []):
        if category == "personality":
            state["exclude_personality"] = True
            state["include_personality"] = False
        elif category == "cognitive":
            state["include_cognitive"] = False
        elif category == "simulation":
            state["include_simulation"] = False

    hint = constraints.get("seniority_hint")
    if hint:
        state["seniority"] = hint
    if constraints.get("reduce_coding"):
        state["reduce_coding"] = True
    if constraints.get("leadership"):
        state["leadership_focus"] = True

    return state


def _detect_role(user_text_lower: str) -> str:
    for pattern, role in SPECIFIC_ROLE_PATTERNS:
        if re.search(pattern, user_text_lower, re.I):
            return role
    if GENERIC_ROLE_PATTERN.search(user_text_lower):
        return "generic technical"
    return ""


def extract_state(messages: list[dict]) -> dict:
    """
    Reconstruct conversational state from full message history.
    Returns structured constraint object.
    """
    state = {
        "role": "",
        "seniority": "",
        "industry": "",
        "language": "",
        "inferred_tags": [],
        "exclude_personality": False,
        "include_personality": False,
        "include_cognitive": False,
        "include_simulation": False,
        "skills": [],
        "conversation_goal": "recommend",
        "needs_clarification": False,
        "turn_count": len(messages),
        "has_recommended": False,
        "explicit_constraints": [],
        "dropped_assessments": [],
        "added_assessments": [],
        "confidence_score": 0,
        "strong_signal_count": 0,
        "ambiguity_score": 0,
        "conversation_archetype": "",
        "conversation_phase": "",
        "missing_hiring_slots": [],
        "policy_action": "recommend",
    }

    full_text = user_text(messages)
    user_text_lower = full_text

    for m in messages:
        if m["role"] == "assistant" and (
            "shl.com" in m["content"]
            or "recommend" in m["content"].lower()
            or "assessment" in m["content"].lower()
        ):
            state["has_recommended"] = True

    for pattern, level in SENIORITY_PATTERNS:
        if re.search(pattern, user_text_lower, re.IGNORECASE):
            state["seniority"] = level
            break

    lang_match = re.search(
        r'\b(english|spanish|french|german|portuguese|chinese|arabic|hindi|dutch|italian|swedish|norwegian|danish|finnish)\b',
        user_text_lower,
        re.I,
    )
    if lang_match:
        lang = lang_match.group(1).capitalize()
        if re.search(r'\b(us|american|usa)\b', user_text_lower, re.I):
            state["language"] = "English (USA)"
        elif re.search(r'\b(uk|british)\b', user_text_lower, re.I):
            state["language"] = "English International"
        elif lang == "Spanish" and re.search(r'\b(latin|mexico|brazil)\b', user_text_lower, re.I):
            state["language"] = "Latin American Spanish"
        else:
            state["language"] = lang
    elif re.search(r'\b(us|american|usa)\b', user_text_lower, re.I):
        state["language"] = "English (USA)"

    for pattern, category in EXCLUDE_PATTERNS:
        if re.search(pattern, user_text_lower, re.IGNORECASE):
            if category == "personality":
                state["exclude_personality"] = True
                state["include_personality"] = False
            elif category == "cognitive":
                state["include_cognitive"] = False

    if re.search(r'\bcognitive tests?\b', user_text_lower, re.I):
        state["include_cognitive"] = True
    if re.search(r'\bpersonality tests?\b', user_text_lower, re.I):
        state["include_personality"] = True
    if re.search(r'\b(simulation|work sample|exercise)\b', user_text_lower, re.I):
        state["include_simulation"] = True
    if re.search(r'\benglish fluent\b', user_text_lower, re.I):
        state["language"] = state.get("language") or "English (USA)"

    for pattern, category in INCLUDE_PATTERNS:
        if re.search(pattern, user_text_lower, re.IGNORECASE):
            if category == "personality":
                state["include_personality"] = True
                state["exclude_personality"] = False
            elif category == "cognitive":
                state["include_cognitive"] = True
            elif category == "simulation":
                state["include_simulation"] = True
            elif category == "communication":
                state["inferred_tags"].append("spoken_language")

    tag_kws = [
        ('safety|hazard|osha|chemical|plant operator|dependab|procedure compliance', 'safety'),
        ('contact cent(er|re)|call cent(er|re)|inbound call', 'contact_center'),
        ('customer service|customer support|customer-facing', 'customer_service'),
        ('executive|cxo|ceo|cto|cfo|c-suite|board level|director', 'executive'),
        ('leadership|leader|strategic|senior leadership', 'leadership'),
        ('sales|revenue|selling|account executive|business development|re-?skill', 'sales'),
        ('graduate|grad scheme|trainee scheme|recent grad|management trainee', 'graduate'),
        ('smart interview|live code|live coding', 'coding'),
        ('\brust\b', 'rust'),
        ('spoken|accent|verbal communication|language screen|bilingual', 'spoken_language'),
        ('excel|word|office|spreadsheet|admin assistant', 'microsoft_office'),
        ('admin assistant|administrative assistant', 'admin'),
        ('healthcare|patient records|hipaa|medical admin', 'healthcare'),
        ('\bhipaa\b', 'hipaa'),
        ('networking|infrastructure|network engineer|high-performance networking', 'networking'),
        (r'\bjava\b', 'java'),
        (r'\bpython\b', 'python'),
        (r'\bsql\b|relational database', 'sql'),
        (r'\baws\b|amazon web service', 'aws'),
        ('docker', 'docker'),
        (r'\bspring\b', 'spring'),
        (r'\bangular\b', 'angular'),
        ('linux', 'linux'),
        ('restructur|talent audit', 'reskilling'),
        (r'\bfull.?stack\b', 'java'),
        (r'\bdata analyt', 'sql'),
        (r'\banalytical reasoning', 'sql'),
        (r'\bwarehouse supervis', 'safety'),
    ]
    for pattern, tag in tag_kws:
        if re.search(pattern, user_text_lower, re.IGNORECASE):
            if tag not in state["inferred_tags"]:
                state["inferred_tags"].append(tag)

    state["role"] = _detect_role(user_text_lower)

    if is_comparison_query(state, messages):
        state["conversation_goal"] = "compare"
    elif is_refinement(state, messages):
        state["conversation_goal"] = "refine"

    last_user = messages[-1]["content"]
    drop_match = re.search(
        r'\b(drop|remove|without|no)\b.*?([A-Z][A-Za-z0-9\s\+]+?)(?:\.|,|$)',
        last_user,
        re.I,
    )
    if drop_match:
        state["dropped_assessments"].append(drop_match.group(2).strip())

    state = merge_constraints(state, extract_constraints(messages))
    state["strong_signal_count"] = count_strong_signals(state, messages)
    state["ambiguity_score"] = ambiguity_score(state, messages)
    state["conversation_archetype"] = infer_archetype(messages, state)
    state["missing_hiring_slots"] = missing_slots(state["conversation_archetype"], messages, state)
    state["conversation_phase"] = conversation_phase(messages, state)
    state["confidence_score"] = state_confidence_score(state, messages)
    state["needs_clarification"] = should_clarify(state, messages)
    state["policy_action"] = resolve_policy_action(state, messages)

    return state


# Role / tag → retrieval vocabulary (expansion only, not recommendations)
ROLE_COMPETENCY_EXPANSIONS = {
    "java": [
        "backend", "spring", "apis", "restful", "sql", "debugging",
        "software engineering", "core java", "object-oriented",
    ],
    "technical hire": [
        "software engineering", "coding", "debugging", "technical skills", "development",
    ],
    "generic technical": [
        "coding", "debugging", "agile", "software development", "programming",
    ],
    "contact center": [
        "communication", "spoken english", "service orientation", "customer service",
        "call simulation", "inbound calls",
    ],
    "customer service": [
        "communication", "customer service", "service orientation", "phone simulation",
    ],
    "sales": [
        "negotiation", "persuasion", "customer interaction", "selling", "sales skills",
    ],
    "graduate scheme": [
        "cognitive", "aptitude", "personality", "graduate scenarios", "situational judgement",
        "verify interactive",
    ],
    "graduate": [
        "cognitive", "aptitude", "personality", "graduate scenarios", "situational judgement",
    ],
    "leadership": [
        "leadership", "management", "executive", "opq", "competency", "strategic",
    ],
    "management": [
        "manager", "supervisor", "leadership", "opq", "cognitive", "competency",
    ],
    "executive": [
        "leadership", "executive", "opq", "personality", "competency", "selection benchmark",
    ],
    "safety": [
        "dependability", "safety", "procedure compliance", "reliability", "manufacturing",
    ],
    "healthcare": [
        "hipaa", "medical terminology", "patient records", "healthcare administration",
    ],
    "admin": [
        "microsoft excel", "microsoft word", "office skills", "administrative",
    ],
    "analytics": [
        "sql", "analytical", "numerical", "data", "reasoning", "verify interactive",
    ],
    "microsoft_office": ["excel", "word", "spreadsheet", "office productivity"],
    "rust": ["linux", "networking", "live coding", "systems programming", "infrastructure"],
    "coding": ["programming", "live coding", "technical interview", "software development"],
}


def extract_jd_features(text: str) -> dict:
    """Heuristic JD / long-form requirement parsing."""
    t = text.lower()
    features = {
        "skills": [],
        "competencies": [],
        "seniority": "",
        "communication": False,
        "leadership": False,
        "is_jd": False,
    }

    if len(t.split()) >= 25 or re.search(
        r"\b(responsibilities|requirements|qualifications|job description|"
        r"what you.?ll do|must have|nice to have|we are looking for)\b",
        t,
    ):
        features["is_jd"] = True

    skill_patterns = [
        (r"\b(java|python|rust|spring|sql|react|node\.?js|angular|aws|docker|kubernetes)\b", None),
        (r"\b(core java|rest api|restful|microservices|full.?stack)\b", None),
        (r"\b(excel|word|office 365|sharepoint)\b", None),
        (r"\b(hipaa|medical terminology)\b", None),
    ]
    for pattern, _ in skill_patterns:
        for m in re.finditer(pattern, t, re.I):
            val = m.group(1).lower() if m.lastindex else m.group(0).lower()
            if val not in features["skills"]:
                features["skills"].append(val)

    if re.search(r"\b(stakeholder|communication|verbal|presentation|interpersonal)\b", t):
        features["communication"] = True
        features["competencies"].append("communication")
    if re.search(r"\b(leadership|manage team|people management|strategic)\b", t):
        features["leadership"] = True
        features["competencies"].append("leadership")
    if re.search(r"\b(problem solving|analytical|critical thinking)\b", t):
        features["competencies"].append("cognitive")
    if re.search(r"\b(customer service|client facing)\b", t):
        features["competencies"].append("customer service")
    if re.search(r"\b(safety|compliance|dependability)\b", t):
        features["competencies"].append("safety")

    for pattern, level in SENIORITY_PATTERNS:
        if re.search(pattern, t, re.I):
            features["seniority"] = level
            break

    return features


def expand_role_terms(state: dict) -> list[str]:
    """Expand role/domain into retrieval vocabulary."""
    terms = []
    role = state.get("role", "")
    if role and role in ROLE_COMPETENCY_EXPANSIONS:
        terms.extend(ROLE_COMPETENCY_EXPANSIONS[role])

    for tag in state.get("inferred_tags", []):
        terms.extend(ROLE_COMPETENCY_EXPANSIONS.get(tag, []))

    for skill in state.get("skills", []):
        terms.append(skill)
        terms.extend(ROLE_COMPETENCY_EXPANSIONS.get(skill, []))

    jd = state.get("jd_features") or {}
    terms.extend(jd.get("skills", []))
    terms.extend(jd.get("competencies", []))

    if state.get("include_personality"):
        terms.extend(["personality", "opq", "behavioral"])
    if state.get("include_cognitive"):
        terms.extend(["cognitive", "ability", "verify interactive", "reasoning"])
    if state.get("include_simulation"):
        terms.extend(["simulation", "exercise", "work sample"])
    if state.get("language"):
        terms.append(state["language"])

    seen = set()
    unique = []
    for term in terms:
        key = term.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(term)
    return unique


def build_search_query(state: dict, messages: list[dict]) -> str:
    """
    Conversation-aware retrieval query: full history + expansions + JD features.
    """
    parts = []

    if state.get("role"):
        parts.append(state["role"])
    if state.get("seniority"):
        parts.append(state["seniority"])
    if state.get("industry"):
        parts.append(state["industry"])
    if state.get("language"):
        parts.append(state["language"])

    parts.extend(state.get("inferred_tags", []))
    parts.extend(expand_role_terms(state))
    parts.extend(state.get("search_expansion", []))

    for tag in state.get("inferred_tags", []):
        parts.append(TAG_QUERY_EXPANSIONS.get(tag, ""))

    # All user turns (preserve multi-turn context)
    for m in messages:
        if m["role"] == "user":
            parts.append(m["content"])

    if state.get("reduce_coding"):
        parts.append("personality communication cognitive simulation non-coding")

    if state.get("leadership_focus"):
        parts.append("leadership opq managerial competency executive")

    return " ".join(p for p in parts if p)


def reconstruct_state(messages: list[dict]) -> dict:
    """
    Deterministic full-history state reconstruction (stateless API).
    """
    state = extract_state(messages)
    full = user_text(messages)

    jd = extract_jd_features(full)
    state["jd_features"] = jd
    if jd.get("seniority") and not state.get("seniority"):
        state["seniority"] = jd["seniority"]
    if jd.get("communication"):
        state["inferred_tags"] = list(dict.fromkeys(
            state.get("inferred_tags", []) + ["spoken_language"]
        ))
    if jd.get("leadership"):
        state["leadership_focus"] = True
        if "leadership" not in state["inferred_tags"]:
            state["inferred_tags"].append("leadership")

    for skill in jd.get("skills", []):
        if skill not in state["skills"]:
            state["skills"].append(skill)
        if skill not in state["inferred_tags"] and skill in SPECIFIC_TECH_TAGS:
            state["inferred_tags"].append(skill)

    constraints = extract_constraints(messages)
    if re.search(r"\b(less coding|reduce coding|not coding|without coding|non.?technical)\b", full, re.I):
        state["reduce_coding"] = True
    if re.search(r"\b(leadership|managerial|management potential)\b", full, re.I):
        state["leadership_focus"] = True
    if constraints.get("communication"):
        state["communication_focus"] = True
    if constraints.get("reduce_coding"):
        state["reduce_coding"] = True

    state["search_expansion"] = expand_role_terms(state)
    state["strong_signal_count"] = count_strong_signals(state, messages)
    state["ambiguity_score"] = ambiguity_score(state, messages)
    state["conversation_archetype"] = infer_archetype(messages, state)
    state["missing_hiring_slots"] = missing_slots(state["conversation_archetype"], messages, state)
    state["conversation_phase"] = conversation_phase(messages, state)
    state["confidence_score"] = state_confidence_score(state, messages)
    state["needs_clarification"] = should_clarify(state, messages)
    state["policy_action"] = resolve_policy_action(state, messages)
    state["_messages"] = messages

    return state


def resolve_policy_action(state: dict, messages: list[dict]) -> str:
    """Route conversation: refuse | refine | compare | clarify | recommend."""
    if is_offtopic(last_user_text(messages)):
        return "refuse"
    if is_refinement(state, messages):
        return "refine"
    if is_comparison_query(state, messages):
        return "compare"
    if should_clarify(state, messages):
        return "clarify"
    if should_recommend(state, messages):
        return "recommend"
    return "clarify"


def build_retrieval_query(messages: list[dict], state: dict) -> str:
    """Backward-compatible alias for conversation-aware search query."""
    if not state.get("search_expansion"):
        state["search_expansion"] = expand_role_terms(state)
    return build_search_query(state, messages)


# Backward-compatible alias used by older call sites
def should_recommend_now(messages: list[dict], state: dict) -> bool:
    return should_recommend(state, messages)
