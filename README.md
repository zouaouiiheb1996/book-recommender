#  AI-Powered Semantic Book Recommender

An AI-powered book recommendation system that combines **semantic search**, **emotion analysis**, and **Large Language Models (LLMs)** to recommend books based on natural language descriptions.

The application retrieves the most relevant books using **OpenAI embeddings** stored in **ChromaDB**, filters them by category and emotional tone, and generates personalized recommendation explanations using **GPT-4o-mini**. An interactive **Gradio** dashboard provides an intuitive user interface, and the entire application is containerized with **Docker**.

---

#  Features

-  Semantic search using vector embeddings
-  Natural language book recommendations
-  Emotion-aware recommendation filtering
-  Category filtering
-  AI-generated recommendation explanations using GPT-4o-mini
-  Interactive Gradio dashboard
-  Dockerized application

---

#  Project Architecture

```text
                        User Query
                             │
                             ▼
             OpenAI Embeddings + ChromaDB
                  (Semantic Search)
                             │
                             ▼
                  Category Filtering
                             │
                             ▼
                 Emotion-Based Ranking
                             │
                             ▼
         GPT-4o-mini Recommendation Generation
                             │
                             ▼
                  Interactive Gradio Dashboard
```

---

#  Project Structure

```text
Book_Recommender/
│
├── app/
│   ├── gradio_app.py
│   └── cover-not-found.jpg
│
├── data/
│   ├── books.csv
│   ├── books_cleaned.csv
│   ├── books_with_categories.csv
│   ├── books_with_emotions.csv
│   ├── tagged_description.txt
│   └── ...
│
├── notebooks/
│   ├── Data_exploration.ipynb
│   ├── Sentiment_analysis.ipynb
│   ├── LLM_prompt.ipynb
│   └── ...
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
├── .env.example
└── README.md
```

---

#  Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Application development |
| Pandas | Data processing |
| NumPy | Numerical computing |
| LangChain | LLM orchestration |
| OpenAI API | Embeddings & explanation generation |
| ChromaDB | Vector database |
| Gradio | Web interface |
| Docker | Containerization |

---

#  Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Book_Recommender.git

cd Book_Recommender
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

#  Environment Variables

Create a file called `.env` in the project root.

Example:

```text
OPENAI_API_KEY=your_openai_api_key
```

---

#  Running the Application

```bash
python app/gradio_app.py
```

Open your browser and visit

```text
http://localhost:7860
```

---

#  Running with Docker

## Build the Docker image

```bash
docker build -t a-book-recommender .
```

## Run the Docker container

```bash
docker run --rm -p 7860:7860 --env-file .env a-book-recommender
```

Open

```text
http://localhost:7860
```

---



#  Screenshots

## Dashboard

> Add a screenshot of your Gradio dashboard here.

Example:

```markdown
![Dashboard](assets/dashboard.png)
```

---




#  Dataset

The recommendation system uses a book dataset containing:

- Book titles
- Authors
- Categories
- Descriptions
- Cover images
- Emotion scores

Book descriptions are converted into vector embeddings and stored in ChromaDB, enabling semantic retrieval based on meaning rather than keyword matching.


---

#  Author

**Iheb Zouaoui**

M.Eng. Computer Engineering


---

#  License

This project is licensed under the MIT License.