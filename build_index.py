"""
Build the FAISS search index for the Sephora product assistant.

This script turns every product into a short text document, converts those
documents into embeddings with the MiniLM model, and saves them in a FAISS
index file called "sephora_products.index".

The index must be built from the SAME products, in the SAME order, that
rag.py uses. That way rag.py can look up a product by its row number.

Run this once before using the assistant:

    python build_index.py
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# 1. Load the Sephora product dataset
# ---------------------------------------------------------

df = pd.read_csv("Dataset/product_info.csv")

# The 12 fields we show to the search model.
rag_columns = [
    "product_id",
    "product_name",
    "brand_name",
    "primary_category",
    "secondary_category",
    "tertiary_category",
    "price_usd",
    "rating",
    "reviews",
    "loves_count",
    "highlights",
    "ingredients",
]

products = df[rag_columns].copy()

print("Number of products:", len(products))


# ---------------------------------------------------------
# 2. Turn each product into a text document
# ---------------------------------------------------------

def safe_text(value):
    """Return a clean string, or "Not available" for missing values."""
    if pd.isna(value):
        return "Not available"
    return str(value)


def create_product_document(row):
    """Build the text that describes one product."""
    return f"""
Product: {safe_text(row['product_name'])}
Brand: {safe_text(row['brand_name'])}
Category: {safe_text(row['primary_category'])}
Subcategory: {safe_text(row['secondary_category'])}
Product Type: {safe_text(row['tertiary_category'])}
Price: ${safe_text(row['price_usd'])}
Rating: {safe_text(row['rating'])}
Reviews: {safe_text(row['reviews'])}
Customer Loves: {safe_text(row['loves_count'])}
Highlights: {safe_text(row['highlights'])}
Ingredients: {safe_text(row['ingredients'])[:1000]}
""".strip()


documents = products.apply(create_product_document, axis=1).tolist()


# ---------------------------------------------------------
# 3. Load the embedding model
# ---------------------------------------------------------

print("Loading embedding model (all-MiniLM-L6-v2)...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------
# 4. Turn the documents into normalised embeddings
# ---------------------------------------------------------

print("Creating embeddings... this can take a minute.")

embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True,
)

embeddings = np.asarray(embeddings, dtype="float32")

print("Embeddings shape:", embeddings.shape)


# ---------------------------------------------------------
# 5. Build and save the FAISS index
# ---------------------------------------------------------

# Because the vectors are normalised, an inner-product index (IndexFlatIP)
# gives the same ranking as cosine similarity.
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, "sephora_products.index")

print("Saved sephora_products.index with", index.ntotal, "products.")
