from pathlib import Path

from debate_v_majority.personas.judge_bank import default_judge_bank_dir


def test_default_judge_bank_dir_uses_repo_global_cache():
    resolved = default_judge_bank_dir(artifacts_dir=Path("ignored"))
    assert resolved == Path(__file__).resolve().parents[1] / "out" / "judge_banks"

