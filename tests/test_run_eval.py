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
