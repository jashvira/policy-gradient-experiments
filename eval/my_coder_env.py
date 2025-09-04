# my_coder_env.py
import verifiers as vf
from pathlib import Path
from train.main import resolve_dataset_path, load_mbpp
from datasets import Dataset


def build_dataset(dataset_split: str = "valid", data_dir: Path | str = Path("/home/jash404/RL_experiments/datasets/mbpp")) -> Dataset:
    """Return HF Dataset with columns: question (str), info (dict)."""
    data_dir = Path(data_dir)
    dataset_path = resolve_dataset_path(dataset_split, data_dir)
    raw = load_mbpp(dataset_path)
    rows = [{"question": item.get("prompt", ""), "info": {}} for item in raw]
    return Dataset.from_list(rows)


def build_parser() -> vf.ThinkParser:
    """Construct a ThinkParser to enforce <think> ... </think> formatting."""
    # Extract final code: take content after </think>
    return vf.ThinkParser(extract_fn=lambda x: x)


def build_rubric(parser: vf.Parser) -> vf.Rubric:
    """Compose rubric with minimal, useful rewards for coding tasks."""

    def code_correctness(parser, completion, answer):
        code = parser.parse_answer(completion) or ''
        if not code.strip():
            return 0.0
        try:
            compile(code, '<string>', 'exec')
            return 1.0
        except SyntaxError:
            return 0.0

    def has_code_block(parser, completion):
        code = parser.parse_answer(completion) or ''
        return 1.0 if code and code.strip() else 0.0

    return vf.Rubric(
        funcs=[code_correctness, has_code_block, parser.get_format_reward_func()],
        weights=[1.0, 0.3, 0.2],
        parser=parser,
    )

def load_environment(**kwargs):
    """Load and configure the coding environment."""
    split = kwargs.pop("dataset_split", "valid")
    data_dir = kwargs.pop("data_dir", Path("/home/jash404/RL_experiments/datasets/mbpp"))

    # Load system prompt from mbpp.prompt
    prompt_file = Path("/home/jash404/RL_experiments/mbpp.prompt")
    system_prompt = prompt_file.read_text().strip()

    dataset = build_dataset(split, data_dir)
    parser = build_parser()
    rubric = build_rubric(parser)

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs
    )
