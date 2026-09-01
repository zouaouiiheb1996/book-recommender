import pandas as pd
import pytest
from types import SimpleNamespace

from app.gradio_app import (
    format_authors,
    load_books,
    retrieve_semantic_recommendations,
    generate_explanation,
    recommend_books,
)


# ============================================================
# FAKE DEPENDENCIES
# ============================================================

class FakeVectorDatabase:
    """
    Fake vector database used for unit tests.

    It behaves like the real vector database by providing
    a similarity_search() method, but does not use Chroma
    or OpenAI.
    """

    def similarity_search(self, query, k):
        return [
            SimpleNamespace(
                page_content="1001 A book about adventure"
            ),
            SimpleNamespace(
                page_content="1002 A book about friendship"
            ),
            SimpleNamespace(
                page_content="1003 A book about mystery"
            ),
        ]


class FakeLLM:
    """
    Fake LLM used for unit tests.

    This prevents tests from calling the real OpenAI API.
    """

    def invoke(self, prompt):
        return SimpleNamespace(
            content="This is a good recommendation."
        )


def create_test_books():
    """
    Create a small fake dataset for testing.
    """

    return pd.DataFrame({
        "isbn13": [1001, 1002, 1003],

        "title": [
            "Adventure Book",
            "Friendship Book",
            "Mystery Book"
        ],

        "authors": [
            "Author A",
            "Author B",
            "Author C"
        ],

        "description": [
            "An adventure story",
            "A story about friendship",
            "A mysterious story"
        ],

        "thumbnail": [
            "image1.jpg",
            "image2.jpg",
            "image3.jpg"
        ],

        "large_thumbnail": [
            "image1_large.jpg",
            "image2_large.jpg",
            "image3_large.jpg"
        ],

        "simple_categories": [
            "Fiction",
            "Romance",
            "Fiction"
        ],

        "joy": [
            0.9,
            0.4,
            0.2
        ],

        "surprise": [
            0.2,
            0.8,
            0.5
        ],

        "anger": [
            0.1,
            0.2,
            0.7
        ],

        "fear": [
            0.3,
            0.4,
            0.9
        ],

        "sadness": [
            0.2,
            0.8,
            0.3
        ],
    })


# ============================================================
# TESTS: format_authors()
# ============================================================

def test_format_authors_single():
    result = format_authors(
        "George Orwell"
    )

    assert result == "George Orwell"


def test_format_authors_two():
    result = format_authors(
        "George Orwell;Aldous Huxley"
    )

    assert result == "George Orwell and Aldous Huxley"


def test_format_authors_multiple():
    result = format_authors(
        "Author A;Author B;Author C"
    )

    assert result == (
        "Author A, Author B, and Author C"
    )


# ============================================================
# TESTS: load_books()
# ============================================================

def test_load_books_returns_dataframe():
    books = load_books()

    assert books is not None
    assert isinstance(books, pd.DataFrame)
    assert len(books) > 0


def test_load_books_has_required_columns():
    books = load_books()

    required_columns = [
        "isbn13",
        "title",
        "authors",
        "description",
        "thumbnail",
        "large_thumbnail",
        "simple_categories",
        "joy",
        "surprise",
        "anger",
        "fear",
        "sadness",
    ]

    for column in required_columns:
        assert column in books.columns


# ============================================================
# TESTS: retrieve_semantic_recommendations()
# ============================================================

def test_retrieve_semantic_recommendations_returns_books():
    db = FakeVectorDatabase()
    books = create_test_books()

    results = retrieve_semantic_recommendations(
        query="adventure",
        db_books=db,
        books=books,
        category="All",
        tone="All",
    )

    assert len(results) == 3


def test_retrieve_semantic_recommendations_category_filter():
    db = FakeVectorDatabase()
    books = create_test_books()

    results = retrieve_semantic_recommendations(
        query="story",
        db_books=db,
        books=books,
        category="Fiction",
        tone="All",
    )

    assert len(results) == 2
    assert all(
        results["simple_categories"] == "Fiction"
    )


# ============================================================
# TESTS: emotional tones
# ============================================================

@pytest.mark.parametrize(
    ("tone", "column", "expected_first"),
    [
        ("Happy", "joy", 0.9),
        ("Surprising", "surprise", 0.8),
        ("Angry", "anger", 0.7),
        ("Suspenseful", "fear", 0.9),
        ("Sad", "sadness", 0.8),
    ],
)
def test_retrieve_semantic_recommendations_tone(
    tone,
    column,
    expected_first,
):
    db = FakeVectorDatabase()
    books = create_test_books()

    results = retrieve_semantic_recommendations(
        query="story",
        db_books=db,
        books=books,
        category="All",
        tone=tone,
    )

    assert results.iloc[0][column] == expected_first


# ============================================================
# TESTS: generate_explanation()
# ============================================================

def test_generate_explanation():
    llm = FakeLLM()

    result = generate_explanation(
        query="A book about adventure",
        description=(
            "A young person goes on an exciting journey."
        ),
        llm=llm,
    )

    assert result == (
        "This is a good recommendation."
    )


def test_generate_explanation_uses_llm():
    """
    Verify that generate_explanation() actually calls
    the LLM.
    """

    class TrackingLLM:

        def __init__(self):
            self.called = False
            self.prompt = None

        def invoke(self, prompt):
            self.called = True
            self.prompt = prompt

            return SimpleNamespace(
                content="Test explanation"
            )

    llm = TrackingLLM()

    result = generate_explanation(
        query="A story about friendship",
        description="Two friends go on an adventure.",
        llm=llm,
    )

    assert llm.called is True
    assert "A story about friendship" in llm.prompt
    assert "Two friends go on an adventure." in llm.prompt
    assert result == "Test explanation"


# ============================================================
# TESTS: recommend_books()
# ============================================================

def test_recommend_books_returns_gallery_results():
    db = FakeVectorDatabase()
    books = create_test_books()
    llm = FakeLLM()

    results = recommend_books(
        query="A story about adventure",
        category="All",
        tone="All",
        db_books=db,
        books=books,
        llm=llm,
    )

    assert len(results) == 3

    for image, caption in results:

        assert image is not None
        assert isinstance(caption, str)
        assert len(caption) > 0


def test_recommend_books_contains_title():
    db = FakeVectorDatabase()
    books = create_test_books()
    llm = FakeLLM()

    results = recommend_books(
        query="adventure",
        category="All",
        tone="All",
        db_books=db,
        books=books,
        llm=llm,
    )

    captions = [
        caption
        for image, caption in results
    ]

    assert any(
        "Adventure Book" in caption
        for caption in captions
    )


def test_recommend_books_contains_author():
    db = FakeVectorDatabase()
    books = create_test_books()
    llm = FakeLLM()

    results = recommend_books(
        query="adventure",
        category="All",
        tone="All",
        db_books=db,
        books=books,
        llm=llm,
    )

    captions = [
        caption
        for image, caption in results
    ]

    assert any(
        "Author A" in caption
        for caption in captions
    )


def test_recommend_books_contains_explanation():
    db = FakeVectorDatabase()
    books = create_test_books()
    llm = FakeLLM()

    results = recommend_books(
        query="adventure",
        category="All",
        tone="All",
        db_books=db,
        books=books,
        llm=llm,
    )

    for image, caption in results:

        assert (
            "This is a good recommendation."
            in caption
        )