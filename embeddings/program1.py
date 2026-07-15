# uv add gensim

# cannot be used for LLMs. but can be used for our own tasks such as :
# Sentiment analysis
# Text classification

#Word2Vec : Introduced by Google in 2013.


from gensim.models import Word2Vec

# Sample sentences (tokenized)
sentences = [
    ["the", "cat", "sat", "on", "the", "mat",],
    ["the", "dog", "sat", "on", "the", "log"],
]

# Train Word2Vec model (skip-gram, window=2)
model = Word2Vec(sentences, vector_size=50, window=2, sg=1, min_count=1, epochs=100)

# Get embedding vector for 'cat'
cat_vector = model.wv['cat']

print("Embedding vector for 'cat':\n", cat_vector)

# Find most similar words to 'cat'
similar_words = model.wv.most_similar('cat', topn=3)
print("\nWords most similar to 'cat':", similar_words)

