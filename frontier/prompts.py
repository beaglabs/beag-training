"""Prompt templates for DeepSeek cold start labeling.

Each task type has a structured template. Templates accept:
- Task-specific config (labels, query, fields, custom description)
- Optional few-shot examples (up to 10)
"""

from __future__ import annotations

from typing import Literal

TaskType = Literal["classification", "relevance", "extraction", "code", "custom"]

# ─── Individual task prompts ──────────────────────────────────────────

CLASSIFICATION_SYSTEM = """You are a domain expert classifying documents into predefined categories.

Your task: read each document and assign exactly ONE category from the list below.

Rules:
- Respond with only the category name, nothing else
- If multiple categories could apply, pick the most specific one
- If no category fits well, pick the closest match
- Do not explain your reasoning"""

CLASSIFICATION_USER = """Categories: {labels}

Classify the text below:"""

CLASSIFICATION_FEWSHOT_USER = """Categories: {labels}

Here are some examples:

{few_shot_examples}

Now classify this text:"""


RELEVANCE_SYSTEM = """You are an expert at determining document relevance for specific queries.

Your task: evaluate whether a document is relevant to a given query.

Rules:
- "relevant" means the document directly addresses or contains information about the query
- "not_relevant" means the document is unrelated, only tangentially mentions the topic, or is off-topic
- Respond with exactly "relevant" or "not_relevant"
- Do not explain your reasoning"""

RELEVANCE_USER = """Query: {query}

Document text:"""


EXTRACTION_SYSTEM = """You are an expert at extracting structured information from unstructured text.

Your task: read the text and extract the requested fields. Return ONLY valid JSON.

Rules:
- Return exactly the fields requested, no extra fields
- Use null for any field not found in the text
- Preserve exact values from the text (do not paraphrase)
- Return the JSON object on a single line"""

EXTRACTION_USER = """Extract these fields: {fields}

Text:"""


CODE_SYSTEM = """You are an expert code reviewer analyzing source code for quality and correctness.

Your task: classify each code snippet into exactly one of these categories:

- "clean": Well-structured, follows best practices, no issues
- "bug": Contains a logical error that would produce incorrect behavior
- "vulnerability": Contains a security weakness (injection, exposure, auth bypass, etc.)
- "anti_pattern": Follows a known anti-pattern or bad practice that hurts maintainability

Rules:
- Respond with only the category name, nothing else
- If multiple issues exist, pick the most severe (vulnerability > bug > anti_pattern > clean)
- Consider the language ecosystem's best practices"""

CODE_USER = """Analyze this {language} code:"""

CODE_FEWSHOT_USER = """Analyze {language} code. Here are examples:

{few_shot_examples}

Now analyze this code:"""


CUSTOM_SYSTEM = """You are an expert assistant performing a specialized document analysis task.

{task_description}

Rules:
- Follow the task description exactly
- Be consistent in your output format
- Do not add commentary beyond what is requested"""

CUSTOM_USER = """Text to analyze:"""


# ─── Prompt builders ───────────────────────────────────────────────────

def build_system_prompt(
    task_type: TaskType,
    task_description: str = "",
    labels: list[str] | None = None,
    query: str = "",
    fields: list[str] | None = None,
    language: str = "unknown",
) -> str:
    if task_type == "classification":
        return CLASSIFICATION_SYSTEM

    elif task_type == "relevance":
        return RELEVANCE_SYSTEM

    elif task_type == "extraction":
        return EXTRACTION_SYSTEM

    elif task_type == "code":
        return CODE_SYSTEM

    elif task_type == "custom":
        return CUSTOM_SYSTEM.format(task_description=task_description or "Analyze the following text.")

    return CLASSIFICATION_SYSTEM


def build_user_prompt(
    task_type: TaskType,
    labels: list[str] | None = None,
    query: str = "",
    fields: list[str] | None = None,
    language: str = "unknown",
    few_shot_examples: list[dict] | None = None,
) -> str:
    if task_type == "classification":
        label_str = ", ".join(labels) if labels else "category_a, category_b, category_c"
        if few_shot_examples:
            return CLASSIFICATION_FEWSHOT_USER.format(
                labels=label_str,
                few_shot_examples=format_few_shot(few_shot_examples),
            )
        return CLASSIFICATION_USER.format(labels=label_str)

    elif task_type == "relevance":
        return RELEVANCE_USER.format(query=query or "the topic")

    elif task_type == "extraction":
        field_str = ", ".join(fields) if fields else "key fields"
        return EXTRACTION_USER.format(fields=field_str)

    elif task_type == "code":
        if few_shot_examples:
            return CODE_FEWSHOT_USER.format(
                language=language,
                few_shot_examples=format_few_shot(few_shot_examples),
            )
        return CODE_USER.format(language=language)

    elif task_type == "custom":
        return CUSTOM_USER

    return CLASSIFICATION_USER.format(labels=", ".join(labels or []))


def build_messages(
    task_type: TaskType,
    text: str,
    task_description: str = "",
    labels: list[str] | None = None,
    query: str = "",
    fields: list[str] | None = None,
    language: str = "unknown",
    few_shot_examples: list[dict] | None = None,
) -> list[dict]:
    system = build_system_prompt(
        task_type=task_type,
        task_description=task_description,
        labels=labels,
        query=query,
        fields=fields,
        language=language,
    )
    user_prefix = build_user_prompt(
        task_type=task_type,
        labels=labels,
        query=query,
        fields=fields,
        language=language,
        few_shot_examples=few_shot_examples,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{user_prefix}\n\n{text}"},
    ]


# ─── Few-shot formatting ──────────────────────────────────────────────

def format_few_shot(examples: list[dict]) -> str:
    """Format few-shot examples for injection into a prompt.

    Each example should be a dict with 'text' and 'label' keys.
    """
    lines = []
    for i, ex in enumerate(examples[:10], 1):
        text = ex.get("text", ex.get("input", ""))
        label = ex.get("label", ex.get("output", ""))
        lines.append(f"Example {i}:\nInput: {text}\nLabel: {label}")
    return "\n\n".join(lines)
