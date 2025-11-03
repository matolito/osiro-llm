# Osiro-LLM Recommender System

This project is a Python library for building and evaluating recommender systems that leverage Large Language Models (LLMs). It includes implementations for zero-shot and re-ranking recommendation strategies, along with standard baseline models for comparison.

## Project Structure

```
osiro-llm/
├── osiro_llm/
│   ├── recommenders/   # Recommendation model implementations
│   ├── data/           # Data loading and processing
│   ├── evaluation/     # Evaluation metrics and pipeline
│   └── llm/            # LLM wrappers and prompts
├── notebooks/          # Experiment notebooks
├── tests/              # Tests (not implemented yet)
├── README.md
├── pyproject.toml      # Project configuration and dependencies
└── uv.lock             # Pinned versions for reproducible installs
```

## Setup

1.  **Install `uv`:**

    If you don't have `uv`, you can install it via `pip`:
    ```bash
    pip install uv
    ```
    For other installation methods, see the [official `uv` documentation](https://astral.sh/docs/uv#installation).

2.  **Create and activate a virtual environment:**

    ```bash
    uv venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    uv pip sync
    ```

4.  **Set your Google API Key:**

    This project uses the Google Generative AI API (for models like Gemini). You need to set your API key as an environment variable.

    ```bash
    export GOOGLE_API_KEY='your_api_key_here'
    ```

## Code Formatting

This project uses the `black` code formatter to ensure a consistent code style. Before committing any changes, please format your code by running the following command from the root of the project:

```bash
black .
```

## How to Run

The main entry point for running the model comparison is the Jupyter notebook located in the `notebooks` directory.

1.  **Open and run `notebooks/02_model_comparison.ipynb`:**

    This notebook will:
    - Download the MovieLens 1M dataset.
    - Train all the implemented recommender models.
    - Evaluate the models on a test set.
    - Display a table with the performance metrics (Precision@10, Recall@10, nDCG@10) for each model.
