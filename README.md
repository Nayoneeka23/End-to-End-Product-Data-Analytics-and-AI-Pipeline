# Sephora Product Analytics and AI Assistant

An end-to-end project on a Sephora product catalog: clean the data, explore it
with charts, and answer shopping questions with a small Retrieval-Augmented
Generation (RAG) assistant. Everything runs locally and is shown in one
Streamlit dashboard.

## Pipeline

```
Dataset/product_info.csv  (8,494 products)
        |
        |-- Analytics path -------------------------------------------
        |     clean (drop duplicates + rows missing price/rating/reviews)
        |     -> 8,216 products -> KPIs + Plotly charts (Overview, Categories, Brands)
        |
        |-- AI assistant path ----------------------------------------
              build text document per product (12 fields)
              -> MiniLM embeddings (normalised)
              -> FAISS index (IndexFlatIP, cosine similarity)
              -> search returns 200 candidates
              -> apply brand / price / rating filters -> keep top 5
              -> Mistral (via Ollama) writes an answer from those 5 only
```

## Dataset

- Source: Sephora products from Kaggle (`Dataset/product_info.csv`).
- Raw size: **8,494 products**, 27 columns.
- After cleaning (removing duplicates and rows missing a price, rating, or
  review count): **8,216 products** (278 rows dropped).
- The cleaned data covers **302 brands** and **9 categories**.

## Key findings

All numbers below come from the cleaned data (8,216 products).

1. **Price barely predicts rating.** The correlation between price and rating
   is **0.057** - almost zero. Expensive products are not rated higher.
2. **Skincare and Makeup dominate the catalog.** Skincare has 2,351 products
   and Makeup 2,328, followed by Hair (1,416) and Fragrance (1,339).
3. **Overall ratings are high and prices are modest.** The average rating is
   **4.19 / 5** and the median price is **$35**.
4. **Sephora-exclusive products attract more interest at a lower price.**
   Exclusive products average **34,366 loves** vs **28,488** for non-exclusive,
   even though their median price is lower (**$30** vs **$37**).

## How the assistant works

1. Each product becomes a short text document (name, brand, category, price,
   rating, reviews, highlights, ingredients, etc.).
2. `all-MiniLM-L6-v2` turns every document into a 384-dimension embedding.
   Embeddings are normalised, so an inner-product FAISS index
   (`IndexFlatIP`) ranks them by cosine similarity.
3. For a question, the assistant retrieves the **200** closest products, then
   applies the optional **brand / max price / min rating** filters, and keeps
   the **top 5**.
4. Those 5 products are sent to **Mistral** (running locally through Ollama)
   with a strict prompt that tells it to use only the provided products and to
   invent nothing. If Ollama is not running, the dashboard still shows the 5
   products and a friendly message.

## Evaluation

Run `python evaluate.py`. On 10 test questions and 4 filter tests:

- **Category accuracy: 72.0%** (36 of 50 top-5 results were in the expected
  category).
- **Filter respect: 100%** (every product returned obeyed the brand, price, and
  rating filters in all 4 tests).

The weakest questions were "a long lasting foundation for oily skin" (matched
skincare products for oily skin instead of makeup) and "a set of makeup
brushes" (Tools & Brushes is a tiny category of 50 products). These are kept as
honest results.

## How to run

```
pip install -r requirements.txt
ollama pull mistral
python build_index.py
streamlit run app.py
```

- `pip install` sets up the libraries.
- `ollama pull mistral` downloads the local language model (needed only for the
  AI answer; the charts and product search work without it).
- `python build_index.py` creates `sephora_products.index` once.
- `streamlit run app.py` opens the dashboard.

You can also use the command-line version with `python rag.py`.

## Limitations

1. The AI assistant searches all 8,494 raw products (filters handle the few
   products with missing prices or ratings), while the analytics charts use the
   cleaned 8,216-product table. The two views are built for different purposes.
2. Category accuracy is 72%, not perfect - semantic search sometimes mixes
   related categories (for example skincare vs makeup for "oily skin").
3. The answer quality depends on a local Mistral model; it is not fine-tuned on
   beauty data and can be slow on a laptop without a GPU.
