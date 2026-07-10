import json
import subprocess
import sys
from pathlib import Path


def test_run_eval_end_to_end_produces_report(tmp_path):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text("""
dataset:
  name: smoke-test
  version: "0.0.1"

questions:
  - id: q001
    question: "What is X?"
    category: factual
    expected:
      answer: "X is a thing."
      citation_doc_ids: ["d1"]
""")
    out_dir = tmp_path / "reports"

    result = subprocess.run(
        [sys.executable, "scripts/run_eval.py",
         "--questions", str(questions_path), "--out-dir", str(out_dir),
         "--fixture-pipeline"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["report_version"] == "1"
    assert report["metadata"]["dataset"]["name"] == "smoke-test"
    assert "settings" in report["metadata"]
    assert "nvidia_api_key" not in json.dumps(report["metadata"]["settings"])
    assert len(report["results"]) == 1
    assert report["results"][0]["id"] == "q001"
    assert (out_dir / "report.html").exists()


QUESTIONS_YAML = """
dataset:
  name: smoke-test
  version: "0.0.1"

questions:
  - id: q001
    question: "What is X?"
    category: factual
    expected:
      answer: "X is a thing."
      citation_doc_ids: ["d1"]
"""


def _run_eval(tmp_path, *extra_args):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(QUESTIONS_YAML)
    out_dir = tmp_path / "report"
    repo_root = Path(__file__).resolve().parent.parent
    return subprocess.run(
        [sys.executable, "scripts/run_eval.py", "--fixture-pipeline",
         "--questions", str(questions_path), "--out-dir", str(out_dir),
         "--baseline-dir", str(tmp_path / "baselines"), *extra_args],
        cwd=repo_root, capture_output=True, text=True,
    )


def test_update_then_compare_baseline_exits_zero(tmp_path):
    create = _run_eval(tmp_path, "--update-baseline", "--notes", "initial")
    assert create.returncode == 0, create.stderr
    baseline_file = tmp_path / "baselines" / "main.json"
    assert baseline_file.exists()
    payload = json.loads(baseline_file.read_text())
    assert payload["baseline_version"] == 1
    assert payload["notes"] == "initial"
    assert payload["question_set_hash"]

    comparison = _run_eval(tmp_path, "--compare-baseline")
    assert comparison.returncode == 0, comparison.stderr
    assert "Status" in comparison.stdout


def test_compare_without_baseline_exits_2(tmp_path):
    result = _run_eval(tmp_path, "--compare-baseline")
    assert result.returncode == 2
    assert "--update-baseline" in result.stderr


def test_compare_with_corrupt_baseline_exits_3(tmp_path):
    (tmp_path / "baselines").mkdir()
    (tmp_path / "baselines" / "main.json").write_text("{broken")
    result = _run_eval(tmp_path, "--compare-baseline")
    assert result.returncode == 3


def test_update_and_compare_together_is_usage_error(tmp_path):
    result = _run_eval(tmp_path, "--update-baseline", "--compare-baseline")
    assert result.returncode == 5


def test_compare_with_changed_questions_exits_4(tmp_path):
    create = _run_eval(tmp_path, "--update-baseline")
    assert create.returncode == 0, create.stderr
    # mutate the questions file after baseline creation
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(QUESTIONS_YAML.replace("What is X?", "What is Y?"))
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "scripts/run_eval.py", "--fixture-pipeline",
         "--questions", str(questions_path), "--out-dir", str(tmp_path / "r2"),
         "--baseline-dir", str(tmp_path / "baselines"), "--compare-baseline"],
        cwd=repo_root, capture_output=True, text=True,
    )
    assert result.returncode == 4
