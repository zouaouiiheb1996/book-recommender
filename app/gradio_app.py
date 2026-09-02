"""
Semantic Book Recommender

This application combines semantic search, emotion filtering,
and a Large Language Model (LLM) to recommend books based on
natural language queries.
"""

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
import gradio as gr
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
APP_DIR = BASE_DIR / "app"


def load_books():
    """Load and prepare the processed book dataset."""

    books = pd.read_csv(DATA_DIR / "books_with_emotions.csv")

    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        str(APP_DIR / "cover-not-found.jpg"),
        books["large_thumbnail"],
    )

    return books


def create_llm():
    """Create the LLM used for generating explanations."""

    load_dotenv()
    return ChatOpenAI(model="gpt-4o-mini")


def create_vector_database():
    """Build the Chroma vector database from tagged descriptions."""

    raw_documents = TextLoader(
        str(DATA_DIR / "tagged_description.txt"),
        encoding="utf-8"
    ).load()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=0,
        chunk_overlap=0
    )

    documents = text_splitter.split_documents(raw_documents)

    return Chroma.from_documents(
        documents,
        OpenAIEmbeddings()
    )


def format_authors(authors):
    """Format author names for display."""

    authors_split = authors.split(";")

    if len(authors_split) == 2:
        return f"{authors_split[0]} and {authors_split[1]}"

    elif len(authors_split) > 2:
        return f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"

    else:
        return authors


def retrieve_semantic_recommendations(
        query: str,
        db_books,
        books,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)

    books_list = [
        int(rec.page_content.strip('"').split()[0])
        for rec in recs
    ]

    book_recs = books[
        books["isbn13"].isin(books_list)
    ].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[
            book_recs["simple_categories"] == category
        ].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(
            by="joy",
            ascending=False,
            inplace=True
        )

    elif tone == "Surprising":
        book_recs.sort_values(
            by="surprise",
            ascending=False,
            inplace=True
        )

    elif tone == "Angry":
        book_recs.sort_values(
            by="anger",
            ascending=False,
            inplace=True
        )

    elif tone == "Suspenseful":
        book_recs.sort_values(
            by="fear",
            ascending=False,
            inplace=True
        )

    elif tone == "Sad":
        book_recs.sort_values(
            by="sadness",
            ascending=False,
            inplace=True
        )

    return book_recs


def generate_explanation(query, description, llm):
    """Generate a short explanation for a recommendation."""

    prompt = f"""
    User is looking for: {query}

    Book description:
    {description}

    Explain in 1 short sentence why this book is a good recommendation.
    """

    response = llm.invoke(prompt)

    return response.content


def recommend_books(
        query: str,
        category: str,
        tone: str,
        db_books,
        books,
        llm
):
    """Generate recommendations for the Gradio dashboard."""

    recommendations = retrieve_semantic_recommendations(
        query,
        db_books,
        books,
        category,
        tone
    )

    results = []

    for _, row in recommendations.iterrows():

        explanation = generate_explanation(
            query,
            row["description"],
            llm
        )

        description = row["description"]

        truncated_desc_split = description.split()

        truncated_description = (
            " ".join(truncated_desc_split[:30]) + "..."
        )

        authors_str = format_authors(row["authors"])

        caption = (
            f"{row['title']} by {authors_str}: "
            f"{explanation}"
        )

        results.append(
            (row["large_thumbnail"], caption)
        )

    return results


def create_dashboard(books, db_books, llm):

    categories = [
        "All"
    ] + sorted(
        books["simple_categories"].unique()
    )

    tones = [
        "All",
        "Happy",
        "Surprising",
        "Angry",
        "Suspenseful",
        "Sad"
    ]

    with gr.Blocks(
        theme=gr.themes.Glass()
    ) as dashboard:

        gr.Markdown(
            "# Semantic book recommender::"
        )

        with gr.Row():

            user_query = gr.Textbox(
                label="Please enter a description of a book:",
                placeholder="e.g., A story about forgiveness"
            )

            category_dropdown = gr.Dropdown(
                choices=categories,
                label="Select a category:",
                value="All"
            )

            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="Select an emotional tone:",
                value="All"
            )

            submit_button = gr.Button(
                "Find recommendations"
            )

        gr.Markdown("## Recommendations")

        output = gr.Gallery(
            label="Recommended books",
            columns=8,
            rows=2
        )

        submit_button.click(
            fn=lambda query, category, tone:
                recommend_books(
                    query,
                    category,
                    tone,
                    db_books,
                    books,
                    llm
                ),
            inputs=[
                user_query,
                category_dropdown,
                tone_dropdown
            ],
            outputs=output
        )

    return dashboard


def main():

    books = load_books()

    llm = create_llm()

    db_books = create_vector_database()

    dashboard = create_dashboard(
        books,
        db_books,
        llm
    )

    dashboard.launch(
        server_name="0.0.0.0",
        server_port=7860
    )


if __name__ == "__main__":
    main()