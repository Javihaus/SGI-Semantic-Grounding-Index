"""Tests for sgi.data module."""

import pytest

from sgi.data import TestCase, get_dataset_stats, print_dataset_summary


class TestTestCase:
    """Tests for the TestCase dataclass."""

    def test_create_test_case(self):
        """Should create TestCase with required fields."""
        case = TestCase(
            id="test_1",
            question="What is AI?",
            context="AI stands for Artificial Intelligence.",
            response="AI is Artificial Intelligence.",
            is_grounded=True,
            source="test",
        )
        assert case.id == "test_1"
        assert case.question == "What is AI?"
        assert case.is_grounded is True

    def test_optional_metadata(self):
        """Should support optional metadata field."""
        case = TestCase(
            id="test_1",
            question="Question",
            context="Context",
            response="Response",
            is_grounded=True,
            source="test",
            metadata={"key": "value"},
        )
        assert case.metadata == {"key": "value"}

    def test_default_metadata_none(self):
        """Metadata should default to None."""
        case = TestCase(
            id="test_1",
            question="Question",
            context="Context",
            response="Response",
            is_grounded=True,
            source="test",
        )
        assert case.metadata is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        case = TestCase(
            id="test_1",
            question="Q",
            context="C",
            response="R",
            is_grounded=True,
            source="src",
        )
        d = case.to_dict()
        assert d["id"] == "test_1"
        assert d["question"] == "Q"
        assert d["context"] == "C"
        assert d["response"] == "R"
        assert d["is_grounded"] is True
        assert d["source"] == "src"
        assert "metadata" not in d  # metadata excluded from to_dict


class TestGetDatasetStats:
    """Tests for the get_dataset_stats function."""

    def test_empty_dataset(self):
        """Should handle empty dataset."""
        stats = get_dataset_stats([])
        assert stats["total"] == 0
        assert stats["grounded"] == 0
        assert stats["hallucinated"] == 0
        assert stats["balance"] == 0
        assert stats["sources"] == {}

    def test_balanced_dataset(self, sample_test_cases):
        """Should compute correct stats for balanced dataset."""
        stats = get_dataset_stats(sample_test_cases)
        assert stats["total"] == 4
        assert stats["grounded"] == 2
        assert stats["hallucinated"] == 2
        assert stats["balance"] == 0.5

    def test_all_grounded(self):
        """Should handle all grounded cases."""
        cases = [
            TestCase(
                id=f"test_{i}",
                question="Q",
                context="C",
                response="R",
                is_grounded=True,
                source="test",
            )
            for i in range(5)
        ]
        stats = get_dataset_stats(cases)
        assert stats["grounded"] == 5
        assert stats["hallucinated"] == 0
        assert stats["balance"] == 1.0

    def test_all_hallucinated(self):
        """Should handle all hallucinated cases."""
        cases = [
            TestCase(
                id=f"test_{i}",
                question="Q",
                context="C",
                response="R",
                is_grounded=False,
                source="test",
            )
            for i in range(5)
        ]
        stats = get_dataset_stats(cases)
        assert stats["grounded"] == 0
        assert stats["hallucinated"] == 5
        assert stats["balance"] == 0.0

    def test_multiple_sources(self):
        """Should count sources correctly."""
        cases = [
            TestCase(
                id="1",
                question="Q",
                context="C",
                response="R",
                is_grounded=True,
                source="source_a",
            ),
            TestCase(
                id="2",
                question="Q",
                context="C",
                response="R",
                is_grounded=True,
                source="source_a",
            ),
            TestCase(
                id="3",
                question="Q",
                context="C",
                response="R",
                is_grounded=False,
                source="source_b",
            ),
        ]
        stats = get_dataset_stats(cases)
        assert stats["sources"]["source_a"] == 2
        assert stats["sources"]["source_b"] == 1


class TestPrintDatasetSummary:
    """Tests for the print_dataset_summary function."""

    def test_prints_without_error(self, sample_test_cases, capsys):
        """Should print summary without errors."""
        print_dataset_summary(sample_test_cases, "Test Dataset")
        captured = capsys.readouterr()
        assert "Test Dataset" in captured.out
        assert "Total samples: 4" in captured.out
        assert "Grounded: 2" in captured.out
        assert "Hallucinated: 2" in captured.out

    def test_empty_dataset(self, capsys):
        """Should handle empty dataset."""
        print_dataset_summary([], "Empty Dataset")
        captured = capsys.readouterr()
        assert "Empty Dataset" in captured.out
        assert "Total samples: 0" in captured.out


class TestDataLoaders:
    """Tests for dataset loader functions.

    Note: These tests are marked as skipped by default since they
    require network access to download datasets.
    """

    @pytest.mark.skip(reason="Requires network access and dataset download")
    def test_load_halueval_qa(self):
        """Test loading HaluEval QA dataset."""
        from sgi.data import load_halueval_qa

        cases = load_halueval_qa(max_samples=10)
        assert len(cases) <= 10
        assert all(isinstance(c, TestCase) for c in cases)
        assert all(c.source == "halueval_qa" for c in cases)

    @pytest.mark.skip(reason="Requires network access and dataset download")
    def test_load_halueval_dialogue(self):
        """Test loading HaluEval Dialogue dataset."""
        from sgi.data import load_halueval_dialogue

        cases = load_halueval_dialogue(max_samples=10)
        assert len(cases) <= 10
        assert all(isinstance(c, TestCase) for c in cases)
        assert all(c.source == "halueval_dialogue" for c in cases)

    @pytest.mark.skip(reason="Requires network access and dataset download")
    def test_load_truthfulqa(self):
        """Test loading TruthfulQA dataset."""
        from sgi.data import load_truthfulqa

        cases = load_truthfulqa(max_samples=10)
        assert len(cases) > 0
        assert all(isinstance(c, TestCase) for c in cases)
        assert all(c.source == "truthfulqa" for c in cases)
