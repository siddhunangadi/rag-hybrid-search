from scripts.benchmark import build_benchmark_corpus, run_benchmark


def test_benchmark_retrieval_quality_meets_baseline(tmp_path):
    corpus = build_benchmark_corpus(tmp_path)
    results = run_benchmark(corpus, top_k=3)

    assert results["num_queries"] == 6
    assert results["recall_at_3"] == 1.0
    assert results["mrr"] >= 0.9
