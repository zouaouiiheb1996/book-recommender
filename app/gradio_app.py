"""
Semantic Book Recommender

This application combines semantic search, emotion filtering,
and a Large Language Model (LLM) to recommend books based on
natural language queries.

Workflow
--------
1. Load the processed book dataset.
2. Build a Chroma vector database from tagged book descriptions.
3. Retrieve semantically similar books.
4. Filter recommendations by category and emotional tone.
5. Generate a short LLM-based explanation for each recommendation.
6. Display the results using a Gradio interface.
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


# Improve image quality by requesting higher-resolution thumbnails.
# Replace missing thumbnails with a local placeholder image.
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")
#books = pd.read_csv("books_with_emotions.csv")
books = pd.read_csv(DATA_DIR / "books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    #"cover-not-found.jpg",
    str(APP_DIR / "cover-not-found.jpg"),
    books["large_thumbnail"],
)


# Load the tagged book descriptions and build
# the Chroma vector database used for semantic retrieval.
#raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
raw_documents = TextLoader(str(DATA_DIR / "tagged_description.txt"),encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    """
    Retrieve books that are semantically similar to the
    user's query and optionally filter them by category
    and emotional tone.

    Parameters
    ----------
    query : str
        Natural language search query.

    category : str
        Selected book category.

    tone : str
        Desired emotional tone.

    initial_top_k : int
        Number of semantic search results retrieved
        before filtering.

    final_top_k : int
        Maximum number of books returned.

    Returns
    -------
    pandas.DataFrame
        Recommended books.
    """

    # Retrieve the most semantically similar book descriptions from the vector database.
    recs = db_books.similarity_search(query, k=initial_top_k)
    # Extract the ISBN of every retrieved document.
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    # Retrieve the corresponding books from the dataset.
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def generate_explanation(query, description):
    """
    Generate a short explanation describing why
    the recommended book matches the user's request.
    """

    # Create the prompt sent to the language model.
    prompt = f"""
    User is looking for: {query}

    Book description:
    {description}

    Explain in 1 short sentence why this book is a good recommendation.
    """
    # Generate the recommendation explanation.
    response = llm.invoke(prompt)
    return response.content

def recommend_books(
        query: str,
        category: str,
        tone: str
):
    """
    Generate the recommendations displayed
    in the Gradio dashboard.
    """

    # Retrieve books matching the user's query.
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    # Process every recommended book.
    for _, row in recommendations.iterrows():
        # Generate an explanation using the LLM.
        explanation = generate_explanation(query, row["description"])
        # Create a shortened version of the book description
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        # Format the author names for display.
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]


        caption = f"{row['title']} by {authors_str}: {explanation}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Build the interactive dashboard.
with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender") # Dashboard title.
    # User input controls.
    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")
    # Recommendation gallery.
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)
    # Connect the search button to the recommendation pipeline.
    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch(server_name="0.0.0.0", server_port=7860)