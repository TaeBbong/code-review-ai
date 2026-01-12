"""
Tests for symbol_definition.py
"""
import asyncio
import tempfile
from pathlib import Path

import pytest

from backend.domain.tools.symbol_definition import (
    find_symbol_definition,
    get_definitions_for_symbols,
)


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a sample repository structure for testing"""
    # Python file with function and class
    py_file = tmp_path / "utils.py"
    py_file.write_text("""
def calculate_total(items: list) -> int:
    \"\"\"Calculate total price of items\"\"\"
    total = 0
    for item in items:
        total += item.price
    return total


class UserService:
    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: int):
        return self.db.find(user_id)

    def create_user(self, name: str):
        return self.db.insert({"name": name})
""")

    # TypeScript file
    ts_file = tmp_path / "api.ts"
    ts_file.write_text("""
export function fetchUsers(): Promise<User[]> {
    return fetch('/api/users').then(r => r.json());
}

export class ApiClient {
    private baseUrl: string;

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
    }

    async get(path: string) {
        return fetch(this.baseUrl + path);
    }
}

export const processData = async (data: any) => {
    return data.map(transform);
};
""")

    return tmp_path


@pytest.mark.asyncio
async def test_find_python_function(sample_repo: Path):
    """Test finding a Python function definition"""
    result = await find_symbol_definition(
        repo_root=str(sample_repo),
        symbol="calculate_total",
    )

    assert result is not None
    assert result.symbol == "calculate_total"
    assert result.kind == "function"
    assert "utils.py" in result.file_path
    assert "def calculate_total" in result.body


@pytest.mark.asyncio
async def test_find_python_class(sample_repo: Path):
    """Test finding a Python class definition"""
    result = await find_symbol_definition(
        repo_root=str(sample_repo),
        symbol="UserService",
    )

    assert result is not None
    assert result.symbol == "UserService"
    assert result.kind == "class"
    assert "class UserService" in result.body


@pytest.mark.asyncio
async def test_find_typescript_function(sample_repo: Path):
    """Test finding a TypeScript function definition"""
    result = await find_symbol_definition(
        repo_root=str(sample_repo),
        symbol="fetchUsers",
    )

    assert result is not None
    assert result.symbol == "fetchUsers"
    assert result.kind == "function"


@pytest.mark.asyncio
async def test_find_typescript_class(sample_repo: Path):
    """Test finding a TypeScript class definition"""
    result = await find_symbol_definition(
        repo_root=str(sample_repo),
        symbol="ApiClient",
    )

    assert result is not None
    assert result.symbol == "ApiClient"
    assert result.kind == "class"


@pytest.mark.asyncio
async def test_symbol_not_found(sample_repo: Path):
    """Test behavior when symbol doesn't exist"""
    result = await find_symbol_definition(
        repo_root=str(sample_repo),
        symbol="NonExistentSymbol",
    )

    assert result is None


@pytest.mark.asyncio
async def test_get_multiple_definitions(sample_repo: Path):
    """Test getting multiple definitions at once"""
    results = await get_definitions_for_symbols(
        repo_root=str(sample_repo),
        symbols=["calculate_total", "UserService", "fetchUsers"],
        max_definitions=5,
    )

    assert len(results) >= 2  # Should find at least some
    symbols_found = {r.symbol for r in results}
    assert "calculate_total" in symbols_found or "UserService" in symbols_found


@pytest.mark.asyncio
async def test_exclude_file(sample_repo: Path):
    """Test that exclude_file parameter works"""
    utils_path = str(sample_repo / "utils.py")

    result = await find_symbol_definition(
        repo_root=str(sample_repo),
        symbol="calculate_total",
        exclude_file=utils_path,
    )

    # Should not find it since it's in the excluded file
    assert result is None
