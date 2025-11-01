import asyncio
import functools
import math
from typing import Dict, List, Optional, Tuple
import pandas as pd
import httpx
from openai import AsyncAzureOpenAI
import os
import random
import atexit
from datasets import load_dataset
import re

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

TAXONOMY_CLASSIFICATION_PROMPT = """Read the prompt below and decide which task category it belongs to. Only output a single category letter (A, B, C, D, E, F, or G) without any additional text. For prompts that have objective responses, choose from categories A, B, C, or D. For prompts that have subjective responses, choose from categories E, F, or G.

Prompt: {prompt}

Task Categories:
A — Well-Specified Singular Objective: Task to generate a single verifiable correct answer.
B — Underspecified Singular Objective: Task to generate a single answer for a prompt that has multiple verifiable correct answers.
C — Random Generation Objective: Task to generate a response that involves randomizing over a set of finite options.
D — Problem Solving Objective: Task to generate an answer with reasoning or explanations for a problem with a single verifiable correct answer.
E — Encyclopedia Inquiry Subjective: Task to generate information about real-world societies, traditions, events, or social domains, where there are credible references.
F — Creative Generation Subjective: Task to generate a response that involves creative expression where there are potentially infinite subjective responses.
G — Advice or Opinion Subjective: Task to generate a response that gives advice, opinions, or feedback on specific topics or scenarios.
""".strip()

dataset = load_dataset("parquet", data_files="/checkpoint/ai_society/shomikj/wildchat10k_merged.parquet")
dataset['train'] = dataset['train'].map(lambda row: {"extracted_prompt": row["prompt"][0]["content"] if row["prompt"] else None})
df = dataset['train'].to_pandas()
df['extracted_prompt'] = df['extracted_prompt'].str.lower().str.replace(r'[^a-zA-Z0-9]', '', regex=True)
PROMPT_CATEGORY_LOOKUP = dict(zip(df['extracted_prompt'], df['category_gpt']))

diversity_judge_prefix = (
    "For the given prompt and two responses, determine if the responses are functionally equivalent. "
    "Functional equivalence means a user who has seen one response would find the other response to be redundant.\n\n"
)

diversity_judge_body = (
    "\n\n###\n"
    "Prompt: {prompt}\n\n"
    "Response 1: {response_1}\n\n"
    "Response 2: {response_2}\n"
    "###\n\n"
    "Are the responses functionally equivalent?\n"
)

diversity_judge_suffix = "\nOnly output YES or NO."

df = pd.read_csv("/checkpoint/ai_society/shomikj/eval/prompt_templates.csv")
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
DIVERSITY_JUDGE_PROMPTS = {}
for i,r in df.iterrows():
    DIVERSITY_JUDGE_PROMPTS[r["category"]] = (
        diversity_judge_prefix + r["functional_equivalence_definition"] + diversity_judge_body + r["diversity_judge_options"] + diversity_judge_suffix
    )


MAX_RETRIES = 3
RETRY_DELAY = 1.0

GPT_QUERIER = AsyncAzureOpenAI(
            api_version="2024-06-01",
            api_key=os.environ["OPEN_AI_TOKEN"],
            azure_endpoint=os.environ["OPEN_AI_ENDPOINT"]
)

PROMPT_PREFIX = "system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\n"

# ──────────────────────────────────────────────────────────────────────────────
# VERY CHEAP LOCAL CHECK FOR TINY ANSWERS
# ──────────────────────────────────────────────────────────────────────────────

def maybe_test_equality(r0: str, r1: str) -> Optional[bool]:
    """If both responses ≤ 5 tokens, decide by exact unigram overlap."""
    u0, u1 = r0.strip().lower().split(), r1.strip().lower().split()
    max_len = max(len(u0), len(u1))
    if max_len <= 5:
        return len(set(u0) & set(u1)) * 2 >= max_len
    return None

# ──────────────────────────────────────────────────────────────────────────────
# REMOTE JUDGE CALL
# ──────────────────────────────────────────────────────────────────────────────


async def remote_judge_equivalent(prompt:str, r0: str, r1: str) -> bool:
    """Return True if the LM judge says the responses are equivalent."""
    prompt = prompt.replace(PROMPT_PREFIX, "").replace("assistant\n\n", "").strip()

    category = ""
    hashed_prompt = re.sub(r'[^a-zA-Z0-9]', '', prompt.lower())
    if hashed_prompt in PROMPT_CATEGORY_LOOKUP:
        category = PROMPT_CATEGORY_LOOKUP[hashed_prompt]
    else:
        for attempt in range(MAX_RETRIES):
            try:
                messages = [{"role": "user", "content": TAXONOMY_CLASSIFICATION_PROMPT.format(prompt=prompt)}]
                completion = await GPT_QUERIER.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=10,
                    top_p = 1.0,
                )
                response = completion.choices[0].message.content.strip()
                category = response.strip().upper()
                if category in {"A", "B", "C", "D", "E", "F", "G"}:
                    PROMPT_CATEGORY_LOOKUP[hashed_prompt] = category
                    break
            except Exception as e:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"{type(e).__name__} encountered with querying LLM-judge. Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
    if category not in {"A", "B", "C", "D", "E", "F", "G"}:
        print(f"Unexpected category '{category}' returned from LLM-classifier.")
        category = "F"  # default to creative writing

    judge_prompt = DIVERSITY_JUDGE_PROMPTS[category].format(prompt=prompt, response_1=r0.strip(), response_2=r1.strip())

    messages = [{"role": "user", "content": judge_prompt}]

    for attempt in range(MAX_RETRIES):
        try:
            completion = await GPT_QUERIER.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=10,
                top_p = 1.0,
            )
            response = completion.choices[0].message.content.strip()
            response = response.strip().lower()
            if response not in {"yes", "no"}:
                raise ValueError(f"Unexpected response from diversity LLM-judge: '{response}'")

            return response == "yes"  # judge says responses functionally equivalent
        except Exception as e:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"{type(e).__name__} encountered with querying LLM-judge. Retrying in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
    
    # Should never reach here
    print("Max retries exceeded for remote judge call.")
    return False

# ──────────────────────────────────────────────────────────────────────────────
# EQUIVALENCE PREDICATE (local heuristic → remote judge)
# ──────────────────────────────────────────────────────────────────────────────
async def equivalence_check(
    prompt: str,
    r0: str,
    r1: str,
) -> bool:
    eq = maybe_test_equality(r0, r1)
    if eq is not None:
        return eq
    return await remote_judge_equivalent(prompt, r0, r1)

# ──────────────────────────────────────────────────────────────────────────────
# UNION‑FIND PARTITIONING FOR ONE UID, RUN ON A GIVEN SERVER
# ──────────────────────────────────────────────────────────────────────────────
async def process_uid_partition(
    uid: str,
    responses: List[str],
    indices: List[int],
    prompt: str,
    LOCAL_SEM,
) -> Tuple[str, List[List[int]]]:
    n = len(responses)
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    async def check(i, j):
        async with LOCAL_SEM:  # acquire semaphore slot
            same = await equivalence_check(prompt, responses[i], responses[j])
            return i, j, same

    tasks = [check(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j, same in await asyncio.gather(*tasks):
        if same:
            union(i, j)

    groups: Dict[int, List[int]] = {}
    for k in range(n):
        root = find(k)
        groups.setdefault(root, []).append(indices[k])

    return uid, list(groups.values())

# ──────────────────────────────────────────────────────────────────────────────
# ASYNC FAN‑OUT ACROSS UID BLOCKS & SERVERS
# ──────────────────────────────────────────────────────────────────────────────
async def partition_async(
    solution_str: List[str],
    uid: List[str],
    prompts: List[str],
) -> Dict[str, List[List[int]]]:
    # Bucket answers by UID
    by_uid_resp, by_uid_idx, by_uid_prompt = {}, {}, {}
    for idx, (ans, u, prm) in enumerate(zip(solution_str, uid, prompts)):
        by_uid_resp.setdefault(u, []).append(ans)
        by_uid_idx.setdefault(u, []).append(idx)
        by_uid_prompt.setdefault(u, prm)

    tasks = []
    LOCAL_SEM = asyncio.Semaphore(16)
    for n, (u, resps) in enumerate(by_uid_resp.items()):
        tasks.append(
            asyncio.create_task(
                process_uid_partition(u, resps, by_uid_idx[u], by_uid_prompt[u], LOCAL_SEM)
            )
        )

    results = await asyncio.gather(*tasks)
    return {u: parts for u, parts in results}

# ──────────────────────────────────────────────────────────────────────────────
# SYNC WRAPPER – REWARD = distinct_count / (TOTAL_RESPONSES‑1)
# ──────────────────────────────────────────────────────────────────────────────

def partition(**kwargs) -> List[float]:
    solution_str: List[str] = kwargs.get("solution_str", [])
    uid: List[str] = kwargs.get("uid")
    prompts: List[str] = kwargs.get("prompts", [])

    if uid is None:
        raise ValueError("uid is required for partition reward function")

    uid_parts = asyncio.run(partition_async(solution_str, uid, prompts))

    uid_to_indices: Dict[str, List[int]] = {}
    for i, u in enumerate(uid):
        uid_to_indices.setdefault(u, []).append(i)

    rewards = [0.0] * len(solution_str)
    for u, parts in uid_parts.items():
        total_for_uid = len(uid_to_indices[u])
        denom = max(total_for_uid - 1, 1)  # avoid ÷0 when only 1 answer
        for grp in parts:
            distinct = total_for_uid - len(grp)
            for idx in grp:
                rewards[idx] = distinct / denom

    assert len(prompts) == len(rewards)
    for i, prompt in enumerate(prompts):
        prompt = prompt.replace(PROMPT_PREFIX, "").replace("assistant\n\n", "").strip()
        hashed_prompt = re.sub(r'[^a-zA-Z0-9]', '', prompt.lower())
        if PROMPT_CATEGORY_LOOKUP[hashed_prompt] == "A":
            rewards[i] = 1.0 - rewards[i]

    return rewards


def partition_sigmoid(**kwargs) -> List[float]:
    """Sigmoid‑scaled version of `partition`."""
    return [1 / (1 + math.exp(-r)) for r in partition(**kwargs)]

@atexit.register
def close_gpt_client():
    try:
        asyncio.run(GPT_QUERIER.aclose())
    except Exception:
        pass