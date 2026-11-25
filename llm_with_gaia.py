from dotenv import find_dotenv, load_dotenv
from modelscope import MsDataset

# litellm only auto-loads .env from the process directory, so load the repo-root
# .env explicitly (this cell lives in notebooks/ch02).
load_dotenv(find_dotenv(usecwd=True))

# 方案一：MsDataset 加载（ModelScope 镜像，无需 Hugging Face 授权）
try:
    level1_problems = MsDataset.load(
        'gaia-benchmark/GAIA', subset_name='2023_level1', split='validation'
    )
except Exception as e:
    print(f"MsDataset 加载失败: {e}")
    # 方案二：Hugging Face 加载（需要 GAIA 访问权限和 HF_TOKEN）
    from datasets import load_dataset

    level1_problems = load_dataset(
        'gaia-benchmark/GAIA', '2023_level1', split='validation'
    )

print(f"Number of Level 1 problems: {len(level1_problems)}")


import asyncio
from pydantic import BaseModel, model_validator
from litellm import acompletion
from tqdm.asyncio import tqdm_asyncio

# Pydantic model for GAIA responses
class GaiaOutput(BaseModel):
    is_solvable: bool = False
    unsolvable_reason: str = ""
    final_answer: str = ""

    @model_validator(mode="before")
    @classmethod
    def _none_to_empty_string(cls, data):
        # DeepSeek's json_object mode does not enforce the schema and often
        # returns null for the field it does not use; normalise those to "".
        if isinstance(data, dict):
            data = dict(data)
            for field in ("unsolvable_reason", "final_answer"):
                if data.get(field) is None:
                    data[field] = ""
            if data.get("is_solvable") is None:
                data["is_solvable"] = False
        return data


PROVIDER_SEMAPHORES = {
"deepseek": asyncio.Semaphore(10),
}
def get_provider(model: str) -> str:
    """Extract provider name from model string."""
    return "deepseek" if model.startswith("deepseek/") else "openai"



import re

def _normalize_answer(text: str | None) -> str:
    """Lower-case, collapse whitespace and drop a trailing period."""
    if text is None:
        return ""
    s = " ".join(str(text).strip().lower().split())
    s = s.rstrip(".")
    # "17,000" -> "17000", but keep commas in real comma-separated lists
    if re.fullmatch(r"-?\d[\d,]*(?:\.\d+)?", s):
        s = s.replace(",", "")
    return s

def is_correct(prediction: str | None, answer: str) -> bool:
    """Check match between prediction and answer after light normalisation."""
    if prediction is None or answer is None:
        return False
    return _normalize_answer(prediction) == _normalize_answer(answer)


gaia_prompt = """You are a general AI assistant. I will ask you a question.
First, determine if you can solve this problem with your current capabilities and set "is_solvable" accordingly.
If you can solve it, set "is_solvable" to true and provide your answer in "final_answer".
If you cannot solve it, set "is_solvable" to false and explain why in "unsolvable_reason".
Your final answer should be a number OR as few words as possible OR a comma-separated list of numbers and/or strings.
If you are asked for a number, don't use a comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g., for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma-separated list, apply the above rules depending on whether the element is a number or a string."""


import re

# DeepSeek only accepts response_format={"type": "json_object"} when the prompt
# contains the word "json", and it does not enforce the schema, so spell out the
# exact shape (and forbid null) in every request.
JSON_INSTRUCTION = (
    "Respond ONLY with a single JSON object with exactly these keys:\n"
    '{"is_solvable": true or false, "unsolvable_reason": "", "final_answer": ""}\n'
    "Always include all three keys. Fields that do not apply must be empty "
    "strings, never null. When is_solvable is true, final_answer must be "
    "non-empty and contain only the answer, with no explanation."
)

def _extract_json(text: str) -> str:
    """Strip markdown code fences and isolate the first {...} block."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    return text

async def solve_problem(
    model: str,
    question: str,
    # Bind the prompt at definition time: if the prompt cell above was not run,
    # this cell fails immediately instead of every task failing silently.
    system_prompt: str = gaia_prompt,
    instruction: str = JSON_INSTRUCTION,
) -> GaiaOutput:
    """Solve a single problem and return structured output."""
    provider = get_provider(model)

    async with PROVIDER_SEMAPHORES[provider]:
        response = await acompletion(
            model=model,
            messages=[
                {"role": "system", "content": gaia_prompt + "\n\n" + JSON_INSTRUCTION},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
            num_retries=2,
        )
        finish_reason = response.choices[0].finish_reason
        content = response.choices[0].message.content

        if finish_reason == "refusal" or not content:
            return GaiaOutput(
                is_solvable=False,
                unsolvable_reason=f"Model refused to answer (finish_reason: {finish_reason})",
                final_answer=""
            )

        try:
            return GaiaOutput.model_validate_json(_extract_json(content))
        except ValueError:
            # Keep the raw text as the prediction instead of losing it to an
            # exception, so a bad JSON response is still debuggable/scorable.
            return GaiaOutput(is_solvable=True, final_answer=content.strip())

async def evaluate_gaia_single(problem: dict, model: str) -> dict:
    """Evaluate a single problem-model pair and return result."""
    try:
        output = await solve_problem(model, problem["Question"])
        return {
            "task_id": problem["task_id"],
            "model": model,
            "correct": is_correct(output.final_answer, problem["Final answer"]),
            "is_solvable": output.is_solvable,
            "prediction": output.final_answer,
            "answer": problem["Final answer"],
            "unsolvable_reason": output.unsolvable_reason,
        }
    except Exception as e:
        return {
            "task_id": problem["task_id"],
            "model": model,
            "correct": False,
            "is_solvable": None,
            "prediction": None,
            "answer": problem["Final answer"],
            "error": str(e),
        }

async def run_experiment(
problems: list[dict],
models: list[str],
) -> dict[str, list]:
    """Evaluate all models on all problems."""
    tasks = [
    evaluate_gaia_single(problem, model)
    for problem in problems
    for model in models
    ]
    all_results = await tqdm_asyncio.gather(*tasks)
    # Group results by model
    results = {model: [] for model in models}
    for result in all_results:
        results[result["model"]].append(result)

    # Surface failures instead of hiding them in the per-row "error" field.
    failed = [r for r in all_results if "error" in r]
    if failed:
        print(
            f"WARNING: {len(failed)}/{len(all_results)} tasks failed; "
            f"first error: {failed[0]['error']}"
        )
    return results

MODELS = [
"deepseek/deepseek-flash"
]
subset = level1_problems.select(range(20))
results = await run_experiment(subset, MODELS)


for model, model_results in results.items():
    correct = sum(1 for r in model_results if r["correct"])
    solvable = sum(1 for r in model_results if r.get("is_solvable"))
    total = len(model_results)
    print(f"{model}: {correct}/{total} correct, {solvable}/{total} judged solvable")
