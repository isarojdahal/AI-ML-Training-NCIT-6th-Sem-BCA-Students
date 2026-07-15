
# uv add sentence-transformers
from sentence_transformers import SentenceTransformer

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2') #384 dimensional dense vector space 

# Define some sentences to embed
sentences = [
    "This is an example sentence",
    "Each sentence is converted",
    "into a vector",
    "Fruits are healthy",
    "Vegetables are good for you"
]

# Generate embeddings for the sentences
embeddings = model.encode(sentences)

# Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print(f"Sentence: {sentence}")
    print(f"Embedding (first 10 dimensions): {embedding[:10]}...")
    print("-" * 20)