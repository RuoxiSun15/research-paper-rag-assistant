"""Run repeatable retrieval and end-to-end RAG evaluations."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from src.evaluation.faithfulness import FaithfulnessEvaluator
from src.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank
from src.retrieval.retriever import AzureSearchRetriever


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load and validate a JSON or JSONL evaluation dataset."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    if dataset_path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in dataset_path.read_text().splitlines()
                if line.strip()]
    else:
        rows = json.loads(dataset_path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluation dataset must be a non-empty JSON list or JSONL file")
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or not str(row.get("question", "")).strip():
            raise ValueError(f"Case {index} must contain a non-empty question")
        relevant = row.get("relevant_chunk_ids")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"Case {index} must contain relevant_chunk_ids")
    return rows


def _case_metrics(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> Dict[str, float]:
    return {
        f"precision_at_{k}": precision_at_k(retrieved_ids, relevant_ids, k),
        f"recall_at_{k}": recall_at_k(retrieved_ids, relevant_ids, k),
        f"reciprocal_rank_at_{k}": reciprocal_rank(retrieved_ids, relevant_ids, k),
    }


def run_evaluation(dataset_path: str, top_k: int = 5,
                   include_faithfulness: bool = False,
                   case_delay_seconds: float = 0.0) -> Dict[str, Any]:
    """Evaluate the configured Azure-backed system against labelled cases."""
    cases = load_dataset(dataset_path)
    retriever = AzureSearchRetriever(top_k=top_k)
    faithfulness = FaithfulnessEvaluator() if include_faithfulness else None
    rag = None
    if include_faithfulness:
        from src.agents.rag_agent import MultiAgentRAG
        rag = MultiAgentRAG(top_k=top_k)

    results = []
    for index, case in enumerate(cases, start=1):
        if index > 1 and case_delay_seconds > 0:
            time.sleep(case_delay_seconds)
        question = case["question"]
        source_file = case.get("source_file") or None
        answer = None
        judge_result = None

        if rag:
            response = rag.query(question, source_file=source_file)
            documents = response.get("retrieved_metadata", [])
            answer = response.get("answer", "")
        else:
            documents = retriever.retrieve(
                question, top_k=top_k, source_file=source_file
            )

        retrieved_ids = [doc.get("id") or doc.get("chunk_id") for doc in documents]
        metrics = _case_metrics(retrieved_ids, case["relevant_chunk_ids"], top_k)
        if faithfulness:
            contexts = [doc.get("content", "") for doc in documents]
            judge_result = faithfulness.evaluate(
                question, answer or "", contexts
            ).to_dict()

        results.append({
            "case_id": case.get("id") or f"case-{index}",
            "question": question,
            "source_file": source_file,
            "relevant_chunk_ids": case["relevant_chunk_ids"],
            "retrieved_chunk_ids": retrieved_ids,
            "metrics": metrics,
            "answer": answer,
            "faithfulness": judge_result,
        })

    metric_names = list(results[0]["metrics"])
    summary = {name: mean(item["metrics"][name] for item in results)
               for name in metric_names}
    summary[f"mrr_at_{top_k}"] = summary.pop(f"reciprocal_rank_at_{top_k}")
    if include_faithfulness:
        summary["mean_faithfulness"] = mean(
            item["faithfulness"]["score"] for item in results
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(dataset_path)),
        "top_k": top_k,
        "case_count": len(results),
        "summary": summary,
        "cases": results,
    }


def save_report(report: Dict[str, Any], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return path
