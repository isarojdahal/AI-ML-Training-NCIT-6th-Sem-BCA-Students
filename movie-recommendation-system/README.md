# AI-powered Movie Recommendation System

This project is a movie recommendation engine that uses SentenceTransformers (BERT embeddings) to perform semantic analysis on movie metadata (genres, keywords, directors, actors, overview) and recommend similar movies using cosine similarity.

It supports both the **IMDb 5000 Movie Dataset** and the **TMDB 5000 Movie Dataset**.

## Setup

First, make sure dependencies are installed (using `uv` or standard pip):

```bash
uv sync
```

## Downloading the IMDb Dataset

To download the IMDb movie dataset (`movie_metadata.csv`), run the included download script:

```bash
uv run python download_imdb.py
```

This will fetch the dataset directly into this directory.

## Usage

You can run the recommendation system using `main.py`.

### 1. Default (IMDb Dataset)

By default, the script loads the IMDb dataset and runs test recommendations for "The Dark Knight Rises" and "Avatar":

```bash
uv run python main.py
```

### 2. Getting Recommendations for a Specific Movie (IMDb)

Specify a movie title using the `--movie` flag:

```bash
uv run python main.py --movie "Spectre"
```

### 3. Using the TMDB Dataset

If you have `tmdb_5000_movies.csv` present in the directory, you can switch the dataset to `tmdb` using the `--dataset` flag:

```bash
uv run python main.py --dataset tmdb
```

Or for a specific movie using TMDB:

```bash
uv run python main.py --dataset tmdb --movie "Avatar"
```

### Options

- `--dataset`: Choose between `imdb` (default) and `tmdb`.
- `--movie`: Target movie title to get recommendations for.
- `--top_n`: Number of recommendations to retrieve (default: `5`).
