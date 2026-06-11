import os
import re
import json
import time
import difflib
from typing import List, Dict
from openai import OpenAI


# =========================
# 1. Basic configuration
# =========================

MODEL_NAME = "gpt-4o-mini"   # You can replace it with your available GPT model
TARGET_POOL_SIZE = 50
BATCH_SIZE = 25
MAX_ROUNDS = 10
NEAR_DUP_THRESHOLD = 0.88

OUTPUT_FILE = "diverse_instruction_pool.jsonl"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# =========================
# 2. Semantic constraints
# =========================

CORE_SEMANTICS = {
    "task_type": "function-level vulnerability binary classification",
    "object": "given source code or function-level code snippet",
    "output_space": "0 for non-vulnerable, 1 for vulnerable"
}

PERTURBATION_FACTORS = {
    "syntactic": [
        "imperative sentence",
        "question form",
        "declarative request",
        "concise command"
    ],
    "lexical": [
        "vulnerability",
        "security flaw",
        "security defect",
        "security risk",
        "unsafe behavior"
    ],
    "tone": [
        "formal",
        "concise",
        "security-expert style",
        "developer-oriented style"
    ]
}

ROLE_PRIOR = (
    "You are a cybersecurity expert who designs task instructions "
    "for function-level vulnerability detection."
)


# =========================
# 3. Prompt construction
# =========================

def build_generation_prompt(n: int) -> str:
    prompt = f"""
You are asked to generate {n} English task instructions for function-level vulnerability detection.

Core semantic constraint:
- Task type: determine whether the given function-level code contains a security vulnerability.
- Analysis object: the given source code or function-level code snippet.
- Output label: 1 means vulnerable, 0 means non-vulnerable.
- The instruction must require binary classification only.

Allowed variation dimensions:
- Syntactic variation: imperative sentence, question form, declarative request, or concise command.
- Lexical variation: use different expressions such as vulnerability, security flaw, security risk, security defect, unsafe behavior.
- Tone variation: formal, concise, developer-oriented, or security-expert style.

Strict constraints:
- Do not ask for vulnerability explanation.
- Do not ask for vulnerability localization.
- Do not ask for repair suggestions.
- Do not ask for severity scoring.
- Do not ask for CWE classification.
- Do not change the output space.
- Each instruction must require the model to output only 0 or 1.

Return only a JSON array of strings.
"""
    return prompt.strip()


# =========================
# 4. GPT-based generation
# =========================

def generate_candidates(n: int) -> List[str]:
    prompt = build_generation_prompt(n)

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=ROLE_PRIOR,
        input=prompt,
        temperature=0.9,
    )

    text = response.output_text.strip()

    try:
        candidates = json.loads(text)
        if not isinstance(candidates, list):
            return []
        return [str(x).strip() for x in candidates]
    except json.JSONDecodeError:
        # Fallback: extract quoted strings if the model does not return valid JSON.
        return re.findall(r'"([^"]+)"', text)


# =========================
# 5. Filtering functions
# =========================

def normalize_instruction(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def format_consistent(text: str) -> bool:
    lower = text.lower()

    object_keywords = [
        "code", "source code", "function", "function-level", "code snippet"
    ]

    label_keywords = [
        "0", "1", "binary", "label"
    ]

    has_object = any(k in lower for k in object_keywords)
    has_label_constraint = any(k in lower for k in label_keywords)

    return has_object and has_label_constraint


def readable(text: str) -> bool:
    words = text.split()

    if len(words) < 8:
        return False

    if len(words) > 60:
        return False

    if text.count(",") > 6:
        return False

    return True


def no_extra_task(text: str) -> bool:
    lower = text.lower()

    forbidden_patterns = [
        "explain",
        "explanation",
        "why",
        "locate",
        "localize",
        "line number",
        "repair",
        "fix",
        "patch",
        "suggest",
        "recommend",
        "severity",
        "score",
        "cwe",
        "root cause"
    ]

    return not any(p in lower for p in forbidden_patterns)


def semantically_aligned(text: str) -> bool:
    lower = text.lower()

    vulnerability_keywords = [
        "vulnerab",
        "security flaw",
        "security defect",
        "security risk",
        "unsafe",
        "weakness"
    ]

    classification_keywords = [
        "determine",
        "detect",
        "judge",
        "classify",
        "decide",
        "identify",
        "check",
        "assess",
        "tell whether"
    ]

    has_vul_semantics = any(k in lower for k in vulnerability_keywords)
    has_classification_semantics = any(k in lower for k in classification_keywords)

    return has_vul_semantics and has_classification_semantics


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def is_near_duplicate(text: str, pool: List[str], threshold: float = NEAR_DUP_THRESHOLD) -> bool:
    for existing in pool:
        if similarity(text, existing) >= threshold:
            return True
    return False


def valid_instruction(text: str, pool: List[str]) -> bool:
    text = normalize_instruction(text)

    if not format_consistent(text):
        return False

    if not readable(text):
        return False

    if not no_extra_task(text):
        return False

    if not semantically_aligned(text):
        return False

    if is_near_duplicate(text, pool):
        return False

    return True


# =========================
# 6. Pool construction
# =========================

def construct_instruction_pool() -> List[Dict]:
    pool: List[str] = []
    records: List[Dict] = []

    for round_id in range(1, MAX_ROUNDS + 1):
        if len(pool) >= TARGET_POOL_SIZE:
            break

        print(f"[Round {round_id}] Current pool size: {len(pool)}")

        candidates = generate_candidates(BATCH_SIZE)

        for cand in candidates:
            cand = normalize_instruction(cand)

            if valid_instruction(cand, pool):
                pool.append(cand)
                records.append({
                    "instruction_id": len(pool),
                    "instruction": cand,
                    "source": "gpt_generation",
                    "semantic_constraint": CORE_SEMANTICS,
                    "perturbation_factors": PERTURBATION_FACTORS
                })

            if len(pool) >= TARGET_POOL_SIZE:
                break

        time.sleep(1)

    return records


# =========================
# 7. Save results
# =========================

def save_jsonl(records: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    records = construct_instruction_pool()
    save_jsonl(records, OUTPUT_FILE)

    print(f"Final instruction pool size: {len(records)}")
    print(f"Saved to: {OUTPUT_FILE}")

    for item in records[:5]:
        print(f"{item['instruction_id']}. {item['instruction']}")