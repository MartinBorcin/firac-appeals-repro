import subprocess
import sys


COMMAND_MODULES = [
    "firac_appeals.cli.prepare_cases",
    "firac_appeals.cli.load_source",
    "firac_appeals.cli.annotate",
    "firac_appeals.cli.split_temporal",
    "firac_appeals.cli.build_tasks",
    "firac_appeals.cli.run_baseline",
    "firac_appeals.cli.run_llm_eval",
    "firac_appeals.cli.score_predictions",
    "firac_appeals.cli.score_summaries",
    "firac_appeals.cli.run_rubric",
    "firac_appeals.cli.compare_rubric",
    "firac_appeals.cli.make_results",
]


def test_cli_help_smoke() -> None:
    for module in COMMAND_MODULES:
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
