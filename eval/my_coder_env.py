# my_coder_env.py
import verifiers as vf

def load_environment(**kwargs):
    """Load and configure the coding environment."""
    # 1. Load dataset - using MBPP for coding tasks
    dataset = vf.load_example_dataset("mbpp", split="test")

    # 2. Configure parser - for code extraction and validation
    parser = vf.CodeParser()  # Extracts code blocks from completion

    # 3. Define reward functions -- can automatically reference:
    # - parser, prompt, completion, answer, state, task, info
    def code_correctness(parser, completion, answer):
        """Check if the generated code is correct."""
        code = parser.parse_code(completion) or ''
        if not code.strip():
            return 0.0

        # Basic syntax validation
        try:
            compile(code, '<string>', 'exec')
            # TODO: Add test case execution here
            return 1.0  # Placeholder - should run actual tests
        except SyntaxError:
            return 0.0

    def has_code_block(parser, completion):
        """Reward for having properly formatted code."""
        code = parser.parse_code(completion)
        return 1.0 if code and code.strip() else 0.0

    # 4. Create rubric
    rubric = vf.Rubric(
        funcs=[code_correctness, has_code_block, parser.get_format_reward_func()],
        weights=[1.0, 0.3, 0.2]
    )

    # 5. Return configured environment
    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt="Write clean, correct Python code. Wrap your solution in ```python code blocks.",
        parser=parser,
        rubric=rubric,
        **kwargs  # Pass through additional arguments
    )
