from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # Use "cuda" if you have a GPU available

sentence = "what's the name of the company and what are the products and services offered by the company?"

embedding = model.encode(sentence)

print("original sentence:", sentence)
print("Embedding shape:", embedding.shape)
print("First 10 elements of embedding:", embedding[:10])
print("length of embedding:", len(embedding))
