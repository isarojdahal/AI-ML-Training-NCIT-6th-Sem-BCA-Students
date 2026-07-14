import argparse
import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def parse_json_column(column_data):
    try:
        # Convert string representation of list to actual Python list of dicts
        genres_list = json.loads(column_data)
        # Extract just the 'name' key from each dict
        return " ".join([item['name'] for item in genres_list])
    except:
        return ""

def load_tmdb_dataset():
    print("Loading TMDB dataset...")
    try:
        df = pd.read_csv('tmdb_5000_movies.csv')
    except FileNotFoundError:
        print("Error: 'tmdb_5000_movies.csv' not found. Please place it in this folder.")
        exit(1)

    required_cols = ['genres', 'keywords', 'overview', 'title']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\n❌ Error: The TMDB dataset file 'tmdb_5000_movies.csv' is missing required columns: {missing_cols}")
        if 'cast' in df.columns and 'crew' in df.columns:
            print("👉 Note: The file appears to be the TMDB credits dataset instead of the movies dataset.")
        print("👉 Please download the actual 'tmdb_5000_movies.csv' file from Kaggle (which contains genres and overviews).\n")
        exit(1)

    print("Cleaning and parsing TMDB metadata features...")
    df['clean_genres'] = df['genres'].apply(parse_json_column)
    df['clean_keywords'] = df['keywords'].apply(parse_json_column)
    df['overview'] = df['overview'].fillna('')
    
    # Create soup
    def create_soup(row):
        return f"Genres: {row['clean_genres']}. Keywords: {row['clean_keywords']}. Summary: {row['overview']}"
    
    df['soup'] = df.apply(create_soup, axis=1)
    df['title_clean'] = df['title'].str.strip()
    
    # Create a nice description/snippet column
    df['snippet'] = df['overview'].apply(lambda x: x[:110] + "..." if len(x) > 110 else x)
    return df


def load_imdb_dataset():
    print("Loading IMDb dataset...")
    try:
        df = pd.read_csv('movie_metadata.csv')
    except FileNotFoundError:
        print("Error: 'movie_metadata.csv' not found. Run 'download_imdb.py' to download it.")
        exit(1)

    print("Cleaning and parsing IMDb metadata features...")
    # Clean titles of \xa0 and spaces
    df['title_clean'] = df['movie_title'].apply(lambda x: str(x).replace('\xa0', '').strip() if pd.notna(x) else '')
    
    def create_soup(row):
        genres = str(row['genres']).replace('|', ' ') if pd.notna(row['genres']) else ''
        keywords = str(row['plot_keywords']).replace('|', ' ') if pd.notna(row['plot_keywords']) else ''
        director = str(row['director_name']) if pd.notna(row['director_name']) else ''
        actor1 = str(row['actor_1_name']) if pd.notna(row['actor_1_name']) else ''
        actor2 = str(row['actor_2_name']) if pd.notna(row['actor_2_name']) else ''
        actor3 = str(row['actor_3_name']) if pd.notna(row['actor_3_name']) else ''
        return f"Genres: {genres}. Keywords: {keywords}. Director: {director}. Actors: {actor1} {actor2} {actor3}"
        
    df['soup'] = df.apply(create_soup, axis=1)
    
    # Create a nice description/snippet column
    def make_snippet(row):
        genres = str(row['genres']).replace('|', ', ') if pd.notna(row['genres']) else 'Unknown'
        director = str(row['director_name']) if pd.notna(row['director_name']) else 'Unknown'
        actors = ", ".join([str(row[a]) for a in ['actor_1_name', 'actor_2_name', 'actor_3_name'] if pd.notna(row[a])])
        return f"Genres: {genres} | Director: {director} | Cast: {actors}"
        
    df['snippet'] = df.apply(make_snippet, axis=1)
    df['title'] = df['title_clean'] # Ensure title column matches target output
    return df

def get_recommendations(df, cosine_sim, movie_title, top_n=5):
    target = movie_title.strip()
    indices = pd.Series(df.index, index=df['title_clean']).to_dict()
    
    if target not in indices:
        print(f"\n❌ Movie '{movie_title}' was not found in the dataset. Check spelling or try another title.")
        return
        
    idx = indices[target]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommended_slices = sim_scores[1:top_n+1]
    
    print(f"\n🎯 Top {top_n} AI Recommendations for '{movie_title}':")
    print("-" * 75)
    for rank, (i, score) in enumerate(recommended_slices, 1):
        print(f" {rank}. {df['title'].iloc[i]} | Confidence: {score:.2%}")
        print(f"    Snippet: {df['snippet'].iloc[i]}")
    print("-" * 75)

def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["imdb", "tmdb"], 
        default="imdb", 
        help="Dataset to use: 'imdb' (default) or 'tmdb'"
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=None,
        help="Get recommendations for a specific movie title"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of recommendations to fetch (default: 5)"
    )
    args = parser.parse_args()
    
    if args.dataset == "imdb":
        df = load_imdb_dataset()
    else:
        df = load_tmdb_dataset()
        
    # Initialize BERT and Generate Embeddings
    print("\nLoading MiniLM BERT model (this may take a few seconds on first run)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Generating semantic vector maps for {len(df)} movies...")
    embeddings = model.encode(df['soup'].tolist(), batch_size=64, show_progress_bar=True)
    
    print("Calculating spatial semantic similarity...")
    cosine_sim = cosine_similarity(embeddings, embeddings)
    
    if args.movie:
        get_recommendations(df, cosine_sim, args.movie, top_n=args.top_n)
    else:
        # Default testing runs
        get_recommendations(df, cosine_sim, 'The Dark Knight Rises', top_n=args.top_n)
        get_recommendations(df, cosine_sim, 'Avatar', top_n=args.top_n)

if __name__ == "__main__":
    main()