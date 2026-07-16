# uv add langchain-google-genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()


def generate_embedding(text: str) -> List[float]:
  
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    result = embeddings.embed_query(text)
    return result

def main():
    # Example usage
    sample_text = "This is an example text to generate embedding."
    
    try:
        # Generate embedding (it generates full sentence embedding)
        embedding = generate_embedding(sample_text)
        # Print the embedding
        print(f"Generated embedding for: {sample_text}")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First few values: {embedding}")
        # The embedding output is of full sentence.
        
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")

if __name__ == "__main__":
    main()