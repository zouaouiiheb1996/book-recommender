# AI-Powered Semantic Book Recommender

An AI-powered book recommendation system that combines **semantic search**, **emotion analysis**, and **Large Language Models (LLMs)** to recommend books based on natural language descriptions.

The application retrieves relevant books using **OpenAI embeddings** stored in **ChromaDB**, filters them by category and emotional tone, and generates personalized recommendation explanations using **GPT-4o-mini**. An interactive **Gradio** dashboard provides the user interface.

The application is containerized with **Docker** and deployed automatically to **AWS EC2** using a **GitHub Actions CI/CD pipeline**.

---

# Features

- Semantic search using vector embeddings
- Natural language book recommendations
- Emotion-aware recommendation filtering
- Category filtering
- AI-generated recommendation explanations using GPT-4o-mini
- Interactive Gradio dashboard
- Dockerized application
- Automated tests with pytest
- Continuous Integration with GitHub Actions
- Docker image vulnerability scanning with Trivy
- Docker images published to GitHub Container Registry (GHCR)
- Continuous Deployment to AWS EC2
- Automated deployment health checks

---

# Project Architecture

## Application Architecture

```text
                         User Query
                             │
                             ▼
                 OpenAI Embeddings + ChromaDB
                       Semantic Search
                             │
                             ▼
                    Category Filtering
                             │
                             ▼
                    Emotion-Based Ranking
                             │
                             ▼
                 GPT-4o-mini Explanation
                             │
                             ▼
                  Gradio Web Application
```

## CI/CD Architecture

```text
                         git push
                            │
                            ▼
                     GitHub Repository
                            │
                            ▼
                    GitHub Actions - CI
                            │
              ┌─────────────┴─────────────┐
              │                           │
           pytest                    Docker Build
              │                           │
              └─────────────┬─────────────┘
                            ▼
                     Trivy Security Scan
                            │
                            ▼
                  GitHub Container Registry
                            │
                            ▼
                    GitHub Actions - CD
                            │
                            ▼
                       SSH to EC2
                            │
                            ▼
                  docker pull latest image
                            │
                            ▼
                 Replace running container
                            │
                            ▼
                     Health Check
                            │
                            ▼
                     Live Application
```

---

# Project Structure

```text
book-recommender/
│
├── app/
│   ├── __init__.py
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
├── tests/
│   └── test_app.py
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
├── .env.example
└── README.md
```

---

# Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Application development |
| Pandas | Data processing |
| NumPy | Numerical computing |
| LangChain | LLM and vector-search orchestration |
| OpenAI API | Embeddings and recommendation explanations |
| ChromaDB | Vector database |
| Gradio | Web interface |
| Pytest | Automated testing |
| Docker | Containerization |
| GitHub Actions | CI/CD automation |
| GitHub Container Registry | Docker image registry |
| Trivy | Container vulnerability scanning |
| AWS EC2 | Application hosting |

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/zouaouiiheb1996/book-recommender.git
cd book-recommender
```

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

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the project root.

Example:

```text
OPENAI_API_KEY=your_openai_api_key
```

Never commit your real `.env` file or API key to GitHub.

A `.env.example` file is included as a template.

---

# Running the Application

Start the application locally:

```bash
python app/gradio_app.py
```

Open your browser:

```text
http://localhost:7860
```

---

# Running Tests

The project uses **pytest** for automated testing.

Install pytest if necessary:

```bash
pip install pytest
```

Run the test suite:

```bash
python -m pytest tests/test_app.py
```

The tests cover core functionality including:

- Author formatting
- Book data loading
- Semantic retrieval
- Category filtering
- Emotion/tone filtering
- LLM explanation generation
- Recommendation output structure

---

# Running with Docker

## Build the Docker image

```bash
docker build -t book-recommender .
```

## Run the Docker container

```bash
docker run --rm -p 7860:7860 --env-file .env book-recommender
```

Open:

```text
http://localhost:7860
```

---

# CI/CD Pipeline

The project uses **GitHub Actions** to automatically test, build, scan, publish, and deploy the application.

## Continuous Integration

When code is pushed to the `main` branch or a pull request is created, GitHub Actions:

1. Checks out the repository
2. Sets up Python 3.12
3. Installs dependencies
4. Runs the pytest test suite
5. Builds the Docker image
6. Runs a Docker smoke test
7. Scans the Docker image with Trivy

## Container Registry

After a successful CI run, the Docker image is published to **GitHub Container Registry (GHCR)**.

```text
ghcr.io/zouaouiiheb1996/book-recommender:latest
```

## Continuous Deployment

After a successful CI run on `main`, the CD workflow:

1. Connects to the AWS EC2 instance through SSH
2. Pulls the latest Docker image from GHCR
3. Stops the previous container
4. Removes the previous container
5. Starts the new container
6. Waits for the application to become ready
7. Performs an HTTP health check
8. Fails the deployment if the application does not become healthy

Therefore, a normal:

```bash
git push origin main
```

can automatically result in an updated live application.

---

# AWS Deployment

The application is hosted in a Docker container on an **AWS EC2** instance.

The EC2 instance provides the server, while Docker runs the application container.

The application is exposed on port `7860`.

Current deployment:

```text
http://18.153.155.151:7860
```

---

# Deployment Flow

```text
Developer
    │
    │ git push origin main
    ▼
GitHub
    │
    ▼
GitHub Actions
    │
    ├── Run tests
    ├── Build Docker image
    ├── Docker smoke test
    └── Trivy vulnerability scan
    │
    ▼
GitHub Container Registry
    │
    ▼
AWS EC2
    │
    ├── Pull latest image
    ├── Stop old container
    ├── Start new container
    └── Health check
    │
    ▼
Live Gradio Application
```

---

# Dataset

The recommendation system uses a book dataset containing:

- Book titles
- Authors
- Categories
- Descriptions
- Cover images
- Emotion scores

Book descriptions are converted into vector embeddings and stored in ChromaDB, enabling semantic retrieval based on meaning rather than simple keyword matching.

---

# Author

**Iheb Zouaoui**

M.Eng. Computer Engineering

---

# License

This project is licensed under the MIT License.
