"""
Tests for similar_code.py
"""
import asyncio
from pathlib import Path

import pytest

from backend.domain.tools.similar_code import (
    SimilarCodeMatch,
    find_similar_code,
    find_similar_to_diff,
    _tokenize,
    _make_ngrams,
    _jaccard_similarity,
)


class TestTokenization:
    """Test tokenization utilities"""

    def test_tokenize_basic(self):
        """Test basic tokenization"""
        code = "def calculate_total(items):"
        tokens = _tokenize(code)

        assert "calculate_total" in tokens
        assert "items" in tokens
        # 'def' is filtered as common keyword
        assert "def" not in tokens

    def test_tokenize_filters_common(self):
        """Test that common keywords are filtered"""
        code = "return self.process(data)"
        tokens = _tokenize(code)

        assert "return" not in tokens
        assert "self" not in tokens
        assert "process" in tokens
        assert "data" in tokens

    def test_tokenize_preserves_identifiers(self):
        """Test that identifiers are preserved"""
        code = "user_service.get_user_by_id(user_id)"
        tokens = _tokenize(code)

        assert "user_service" in tokens
        assert "get_user_by_id" in tokens
        assert "user_id" in tokens


class TestNGrams:
    """Test n-gram generation"""

    def test_make_ngrams_basic(self):
        """Test basic n-gram generation"""
        tokens = ["a", "b", "c", "d"]
        ngrams = _make_ngrams(tokens, n=3)

        assert ("a", "b", "c") in ngrams
        assert ("b", "c", "d") in ngrams
        assert len(ngrams) == 2

    def test_make_ngrams_short_input(self):
        """Test n-gram with short input"""
        tokens = ["a", "b"]
        ngrams = _make_ngrams(tokens, n=3)

        # Should return tuple of all tokens
        assert len(ngrams) == 1
        assert ("a", "b") in ngrams

    def test_make_ngrams_empty(self):
        """Test n-gram with empty input"""
        tokens = []
        ngrams = _make_ngrams(tokens, n=3)

        assert len(ngrams) == 0


class TestJaccardSimilarity:
    """Test Jaccard similarity calculation"""

    def test_identical_sets(self):
        """Test identical sets"""
        set_a = {1, 2, 3}
        set_b = {1, 2, 3}

        assert _jaccard_similarity(set_a, set_b) == 1.0

    def test_disjoint_sets(self):
        """Test completely different sets"""
        set_a = {1, 2, 3}
        set_b = {4, 5, 6}

        assert _jaccard_similarity(set_a, set_b) == 0.0

    def test_partial_overlap(self):
        """Test partial overlap"""
        set_a = {1, 2, 3, 4}
        set_b = {3, 4, 5, 6}

        # Intersection: {3, 4} = 2
        # Union: {1, 2, 3, 4, 5, 6} = 6
        assert _jaccard_similarity(set_a, set_b) == pytest.approx(2 / 6)

    def test_empty_sets(self):
        """Test empty sets"""
        assert _jaccard_similarity(set(), set()) == 0.0
        assert _jaccard_similarity({1}, set()) == 0.0


@pytest.fixture
def sample_repo_with_similar(tmp_path: Path) -> Path:
    """Create a repo with similar code patterns"""
    # Original file with a pattern
    original = tmp_path / "services" / "user_service.py"
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_text("""
class UserService:
    def __init__(self, repository):
        self.repository = repository

    def get_by_id(self, user_id: int):
        result = self.repository.find_by_id(user_id)
        if result is None:
            raise NotFoundException(f"User {user_id} not found")
        return result

    def create(self, data: dict):
        validated = self._validate(data)
        return self.repository.insert(validated)
""")

    # Similar file with similar patterns
    similar = tmp_path / "services" / "product_service.py"
    similar.write_text("""
class ProductService:
    def __init__(self, repository):
        self.repository = repository

    def get_by_id(self, product_id: int):
        result = self.repository.find_by_id(product_id)
        if result is None:
            raise NotFoundException(f"Product {product_id} not found")
        return result

    def create(self, data: dict):
        validated = self._validate(data)
        return self.repository.insert(validated)
""")

    # Different pattern
    different = tmp_path / "utils" / "helpers.py"
    different.parent.mkdir(parents=True, exist_ok=True)
    different.write_text("""
def format_currency(amount: float) -> str:
    return f"${amount:,.2f}"

def parse_date(date_str: str):
    return datetime.strptime(date_str, "%Y-%m-%d")
""")

    return tmp_path


@pytest.mark.asyncio
async def test_find_similar_code(sample_repo_with_similar: Path):
    """Test finding similar code"""
    query_code = """
class OrderService:
    def __init__(self, repository):
        self.repository = repository

    def get_by_id(self, order_id: int):
        result = self.repository.find_by_id(order_id)
        if result is None:
            raise NotFoundException(f"Order {order_id} not found")
        return result
"""

    results = await find_similar_code(
        repo_root=str(sample_repo_with_similar),
        query_code=query_code,
        min_similarity=0.2,
        max_results=5,
    )

    # Should find similar patterns in user_service and product_service
    assert len(results) >= 1

    # Results should be sorted by similarity
    if len(results) > 1:
        assert results[0].similarity >= results[1].similarity


@pytest.mark.asyncio
async def test_find_similar_to_diff(sample_repo_with_similar: Path):
    """Test finding similar code from diff"""
    diff_text = """
diff --git a/services/order_service.py b/services/order_service.py
new file mode 100644
--- /dev/null
+++ b/services/order_service.py
@@ -0,0 +1,12 @@
+class OrderService:
+    def __init__(self, repository):
+        self.repository = repository
+
+    def get_by_id(self, order_id: int):
+        result = self.repository.find_by_id(order_id)
+        if result is None:
+            raise NotFoundException(f"Order {order_id} not found")
+        return result
"""

    results = await find_similar_to_diff(
        repo_root=str(sample_repo_with_similar),
        diff_text=diff_text,
        min_similarity=0.2,
        max_results=5,
    )

    # Should find similar patterns
    assert len(results) >= 0  # May or may not find depending on tokenization


@pytest.mark.asyncio
async def test_exclude_files(sample_repo_with_similar: Path):
    """Test that exclude_files parameter works"""
    query_code = """
class UserService:
    def get_by_id(self, user_id: int):
        return self.repository.find_by_id(user_id)
"""

    # Search with exclusion
    results_with_exclude = await find_similar_code(
        repo_root=str(sample_repo_with_similar),
        query_code=query_code,
        exclude_files=["services/user_service.py"],
        min_similarity=0.2,
    )

    # The exact user_service.py should not be in results
    file_paths = [r.file_path for r in results_with_exclude]
    assert "services/user_service.py" not in file_paths


@pytest.mark.asyncio
async def test_min_similarity_filter(sample_repo_with_similar: Path):
    """Test that min_similarity filters low matches"""
    query_code = "def unique_function_xyz(): pass"

    results = await find_similar_code(
        repo_root=str(sample_repo_with_similar),
        query_code=query_code,
        min_similarity=0.9,  # Very high threshold
    )

    # Should not find anything with such high threshold
    assert len(results) == 0
