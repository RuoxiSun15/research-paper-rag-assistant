from src.evaluation.faithfulness import FaithfulnessEvaluator


def test_faithfulness_parses_fenced_json_and_clamps_score():
    def judge(_prompt):
        return """```json
        {"score": 1.2, "supported_claims": ["A"],
         "unsupported_claims": [], "explanation": "Grounded"}
        ```"""

    result = FaithfulnessEvaluator(judge=judge).evaluate(
        "Question?", "Answer A", ["Passage A"]
    )
    assert result.score == 1.0
    assert result.supported_claims == ["A"]
    assert result.unsupported_claims == []


def test_faithfulness_prompt_treats_context_as_data():
    captured = {}

    def judge(prompt):
        captured["prompt"] = prompt
        return '{"score": 0.5, "supported_claims": [], "unsupported_claims": ["B"], "explanation": "Partial"}'

    FaithfulnessEvaluator(judge=judge).evaluate("Q", "B", ["context"])
    assert "Retrieved context" in captured["prompt"]
    assert "[Passage 1]" in captured["prompt"]
