"""
Data Loading Utilities

Standardized data loading for hallucination detection experiments.
Supports HaluEval and TruthfulQA benchmarks.
"""

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class TestCase:
    """Standardized test case for hallucination detection."""

    id: str
    question: str
    context: str
    response: str
    is_grounded: bool  # True = valid/grounded, False = hallucinated
    source: str
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "response": self.response,
            "is_grounded": self.is_grounded,
            "source": self.source,
        }


def load_halueval_qa(
    max_samples: Optional[int] = None,
    min_response_length: int = 20,
) -> List[TestCase]:
    """
    Load HaluEval QA dataset.

    Args:
        max_samples: Maximum number of samples to load (None = all)
        min_response_length: Minimum response length in characters

    Returns:
        List of TestCase objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    cases: List[TestCase] = []
    for i, item in enumerate(dataset):
        if max_samples and len(cases) >= max_samples:
            break

        # Skip incomplete entries
        if not item.get("knowledge") or not item.get("answer"):
            continue

        # Skip very short responses
        if len(item["answer"]) < min_response_length:
            continue

        # Parse hallucination field (handles multiple formats)
        halluc_val = str(item.get("hallucination", "")).lower().strip()
        is_hallucinated = halluc_val in ["yes", "1", "true"]
        is_valid = halluc_val in ["no", "0", "false"]

        if not is_hallucinated and not is_valid:
            continue

        cases.append(
            TestCase(
                id=f"halueval_qa_{i}",
                question=item["question"],
                context=item["knowledge"],
                response=item["answer"],
                is_grounded=is_valid,
                source="halueval_qa",
            )
        )

    return cases


def load_halueval_dialogue(
    max_samples: Optional[int] = None,
    min_response_length: int = 20,
) -> List[TestCase]:
    """
    Load HaluEval Dialogue dataset.

    Args:
        max_samples: Maximum number of samples to load (None = all)
        min_response_length: Minimum response length in characters

    Returns:
        List of TestCase objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    dataset = load_dataset("pminervini/HaluEval", "dialogue_samples", split="data")

    cases: List[TestCase] = []
    for i, item in enumerate(dataset):
        if max_samples and len(cases) >= max_samples:
            break

        if not item.get("knowledge") or not item.get("response"):
            continue

        if len(item["response"]) < min_response_length:
            continue

        halluc_val = str(item.get("hallucination", "")).lower().strip()
        is_hallucinated = halluc_val in ["yes", "1", "true"]
        is_valid = halluc_val in ["no", "0", "false"]

        if not is_hallucinated and not is_valid:
            continue

        # Dialogue context is the conversation history
        dialogue_history = item.get("dialogue_history", "")

        cases.append(
            TestCase(
                id=f"halueval_dial_{i}",
                question=dialogue_history,
                context=item["knowledge"],
                response=item["response"],
                is_grounded=is_valid,
                source="halueval_dialogue",
            )
        )

    return cases


def load_truthfulqa(
    max_samples: Optional[int] = None,
) -> List[TestCase]:
    """
    Load TruthfulQA dataset.

    Note: TruthfulQA doesn't have explicit context, so we use
    the question itself as context (tests question-response alignment).

    Args:
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        List of TestCase objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    dataset = load_dataset("truthful_qa", "generation", split="validation")

    cases: List[TestCase] = []
    for i, item in enumerate(dataset):
        if max_samples and len(cases) >= max_samples:
            break

        question = item["question"]
        best_answer = item.get("best_answer", "")
        correct_answers = item.get("correct_answers", [])
        incorrect_answers = item.get("incorrect_answers", [])

        # Create grounded case (best/correct answer)
        if best_answer:
            cases.append(
                TestCase(
                    id=f"truthfulqa_{i}_grounded",
                    question=question,
                    context=question,  # Self-grounding for TruthfulQA
                    response=best_answer,
                    is_grounded=True,
                    source="truthfulqa",
                )
            )

        # Create hallucinated cases (incorrect answers)
        for j, incorrect in enumerate(incorrect_answers[:2]):  # Limit to 2 per question
            cases.append(
                TestCase(
                    id=f"truthfulqa_{i}_halluc_{j}",
                    question=question,
                    context=question,
                    response=incorrect,
                    is_grounded=False,
                    source="truthfulqa",
                )
            )

    return cases


def get_dataset_stats(cases: List[TestCase]) -> Dict:
    """Get summary statistics for a dataset."""
    grounded = sum(1 for c in cases if c.is_grounded)
    hallucinated = len(cases) - grounded
    sources: Dict[str, int] = {}
    for c in cases:
        sources[c.source] = sources.get(c.source, 0) + 1

    return {
        "total": len(cases),
        "grounded": grounded,
        "hallucinated": hallucinated,
        "balance": grounded / len(cases) if cases else 0,
        "sources": sources,
    }


def print_dataset_summary(cases: List[TestCase], name: str = "Dataset") -> None:
    """Print summary of loaded dataset."""
    stats = get_dataset_stats(cases)
    print(f"\n{name}:")
    print(f"  Total samples: {stats['total']}")
    print(f"  Grounded: {stats['grounded']} ({stats['balance']*100:.1f}%)")
    print(f"  Hallucinated: {stats['hallucinated']} ({(1-stats['balance'])*100:.1f}%)")
    if len(stats["sources"]) > 1:
        print(f"  Sources: {stats['sources']}")
