"""
Sephora Product Analytics and AI Assistant - Streamlit dashboard.

This app has four tabs:
  1. Overview       - headline numbers about the dataset
  2. Categories & Prices - products per category and price vs rating
  3. Brands         - top established brands and Sephora-exclusive comparison
  4. AI Assistant   - ask a question and get product recommendations

Run it with:

    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------
# 1. Load and clean the data (same steps as analysis.ipynb)
# ---------------------------------------------------------

@st.cache_data
def load_clean_data():
    """Load the raw CSV and apply the same cleaning as the notebook.

    Returns the cleaned table and the raw row count so the Overview tab
    can show how many rows were dropped.
    """
    df = pd.read_csv("Dataset/product_info.csv")
    raw_rows = len(df)

    columns = [
        "product_name",
        "brand_name",
        "primary_category",
        "secondary_category",
        "price_usd",
        "rating",
        "reviews",
        "loves_count",
        "sephora_exclusive",
        "new",
        "online_only",
    ]

    data = df[columns].copy()

    # Remove duplicates and rows missing the key numbers.
    data = data.drop_duplicates()
    data = data.dropna(subset=["price_usd", "rating", "reviews"])

    return data, raw_rows


# ---------------------------------------------------------
# 2. Load the AI assistant (model + FAISS index) only once
# ---------------------------------------------------------

@st.cache_resource
def load_assistant():
    """Import rag.py once so the model and index load a single time.

    rag.py loads the embedding model and the FAISS index when it is
    imported. Caching it here means that heavy work happens only once,
    not every time the page refreshes.
    """
    import rag
    return rag


# ---------------------------------------------------------
# 3. Page setup
# ---------------------------------------------------------

st.set_page_config(page_title="Sephora Product Assistant", layout="wide")

st.title("Sephora Product Analytics and AI Assistant")

data, raw_rows = load_clean_data()
clean_rows = len(data)

tab_overview, tab_categories, tab_brands, tab_ai = st.tabs(
    ["Overview", "Categories & Prices", "Brands", "AI Assistant"]
)


# ---------------------------------------------------------
# 4. Overview tab
# ---------------------------------------------------------

with tab_overview:
    st.header("Overview")

    # Five headline numbers.
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Products", f"{clean_rows:,}")
    col2.metric("Brands", f"{data['brand_name'].nunique():,}")
    col3.metric("Categories", data["primary_category"].nunique())
    col4.metric("Average rating", round(data["rating"].mean(), 2))
    col5.metric("Median price", f"${data['price_usd'].median():.0f}")

    st.write(
        f"Data cleaning: started with {raw_rows:,} products and kept "
        f"{clean_rows:,} after removing duplicates and rows missing a "
        f"price, rating, or review count "
        f"({raw_rows - clean_rows:,} rows dropped)."
    )


# ---------------------------------------------------------
# 5. Categories & Prices tab
# ---------------------------------------------------------

with tab_categories:
    st.header("Categories & Prices")

    # Products per category (bar chart).
    category_counts = (
        data["primary_category"]
        .value_counts()
        .reset_index()
    )
    category_counts.columns = ["primary_category", "products"]

    fig_categories = px.bar(
        category_counts,
        x="primary_category",
        y="products",
        title="Number of Products by Category",
        labels={
            "primary_category": "Category",
            "products": "Number of Products",
        },
    )
    st.plotly_chart(fig_categories, use_container_width=True)

    # Price vs rating (scatter) with the correlation number.
    correlation = data["price_usd"].corr(data["rating"])
    st.write(f"Correlation between price and rating: {correlation:.3f}")

    # Drop the most expensive 1% so the chart is easier to read.
    price_plot = data[
        data["price_usd"] <= data["price_usd"].quantile(0.99)
    ]

    fig_price = px.scatter(
        price_plot,
        x="price_usd",
        y="rating",
        color="primary_category",
        hover_data=["product_name", "brand_name"],
        title="Price vs Customer Rating",
        labels={
            "price_usd": "Price (USD)",
            "rating": "Customer Rating",
            "primary_category": "Category",
        },
        opacity=0.6,
    )
    st.plotly_chart(fig_price, use_container_width=True)


# ---------------------------------------------------------
# 6. Brands tab
# ---------------------------------------------------------

with tab_brands:
    st.header("Brands")

    # Build a per-brand summary.
    brand_summary = (
        data.groupby("brand_name")
        .agg(
            products=("product_name", "count"),
            average_rating=("rating", "mean"),
            total_reviews=("reviews", "sum"),
        )
        .reset_index()
    )

    # "Established" = enough products and enough reviews to trust the rating.
    established_brands = brand_summary[
        (brand_summary["products"] >= 10)
        & (brand_summary["total_reviews"] >= 500)
    ]

    top_brands = (
        established_brands
        .sort_values("average_rating", ascending=False)
        .head(10)
    )

    st.subheader("Top 10 Established Brands by Average Rating")
    st.caption("Established = at least 10 products and 500 total reviews.")

    fig_brands = px.bar(
        top_brands.sort_values("average_rating"),
        x="average_rating",
        y="brand_name",
        orientation="h",
        title="Highest-Rated Established Brands",
        labels={
            "average_rating": "Average Rating",
            "brand_name": "Brand",
        },
    )
    st.plotly_chart(fig_brands, use_container_width=True)

    # Sephora exclusive vs non-exclusive.
    st.subheader("Sephora Exclusive vs Non-Exclusive")

    exclusive_summary = (
        data.groupby("sephora_exclusive")
        .agg(
            products=("product_name", "count"),
            average_rating=("rating", "mean"),
            median_price=("price_usd", "median"),
            average_loves=("loves_count", "mean"),
        )
        .reset_index()
    )

    exclusive_summary["product_type"] = exclusive_summary[
        "sephora_exclusive"
    ].map({0: "Non-Exclusive", 1: "Sephora Exclusive"})

    fig_exclusive = px.bar(
        exclusive_summary,
        x="product_type",
        y="average_loves",
        title="Average Customer Loves: Exclusive vs Non-Exclusive",
        labels={
            "product_type": "",
            "average_loves": "Average Loves",
        },
    )
    st.plotly_chart(fig_exclusive, use_container_width=True)

    st.dataframe(
        exclusive_summary[
            ["product_type", "products", "average_rating",
             "median_price", "average_loves"]
        ],
        use_container_width=True,
    )


# ---------------------------------------------------------
# 7. AI Assistant tab
# ---------------------------------------------------------

with tab_ai:
    st.header("AI Assistant")
    st.write(
        "Ask a question about Sephora products. "
        "The assistant finds the best matches and Mistral explains them."
    )

    query = st.text_input(
        "Your question",
        placeholder="e.g. I have dry skin and need a gentle moisturizer",
    )

    # Optional filters.
    col_a, col_b, col_c = st.columns(3)
    brand = col_a.text_input("Brand (optional)")
    max_price = col_b.number_input(
        "Max price ($, optional)", min_value=0.0, value=0.0, step=5.0
    )
    min_rating = col_c.number_input(
        "Min rating (optional)", min_value=0.0, max_value=5.0,
        value=0.0, step=0.5
    )

    search = st.button("Search")

    if search and query.strip():
        rag = load_assistant()

        # Turn the "0 = off" inputs into real filters or None.
        brand_filter = brand.strip() if brand.strip() else None
        price_filter = max_price if max_price > 0 else None
        rating_filter = min_rating if min_rating > 0 else None

        results = rag.retrieve_products(
            query=query,
            max_price=price_filter,
            min_rating=rating_filter,
            brand=brand_filter,
            final_k=5,
            candidate_k=200,
        )

        if results.empty:
            st.warning(
                "No products matched those filters. Try relaxing the "
                "brand, price, or rating."
            )
        else:
            # Always show the products, even if Ollama is off.
            st.subheader("Top 5 products")
            st.dataframe(
                results[
                    ["product_name", "brand_name", "primary_category",
                     "secondary_category", "price_usd", "rating",
                     "similarity_score"]
                ],
                use_container_width=True,
            )

            # Then try to get Mistral's answer.
            st.subheader("AI recommendation")
            try:
                answer = rag.generate_answer(
                    query=query,
                    retrieved_products=results,
                    max_price=price_filter,
                    min_rating=rating_filter,
                    brand=brand_filter,
                )
                st.write(answer)
            except Exception:
                st.info(
                    "The products above are ready, but the AI explanation "
                    "needs Ollama running locally. Start it with "
                    "`ollama pull mistral` and `ollama serve`, then search "
                    "again."
                )
    elif search:
        st.warning("Please type a question first.")
