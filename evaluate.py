"""
Very simple evaluation of the product search.

It checks two things:

  1. Category accuracy - for a list of test questions, how many of the
     top-5 results fall into the category we would expect.
  2. Filter respect   - when we ask for a max price, min rating, or brand,
     do all returned products actually obey those filters.

Run it with:

    python evaluate.py

(The FAISS index must already exist, so run build_index.py first.)
"""

import rag


# ---------------------------------------------------------
# 1. Test questions and the category we expect
# ---------------------------------------------------------

test_questions = [
    ("A gentle moisturizer for dry skin", "Skincare"),
    ("A face cleanser for sensitive skin", "Skincare"),
    ("An anti aging eye cream", "Skincare"),
    ("A long lasting foundation for oily skin", "Makeup"),
    ("A bright red lipstick", "Makeup"),
    ("Waterproof mascara for volume", "Makeup"),
    ("A shampoo for an oily scalp", "Hair"),
    ("A conditioner for damaged hair", "Hair"),
    ("A fresh floral perfume", "Fragrance"),
    ("A set of makeup brushes", "Tools & Brushes"),
]


# ---------------------------------------------------------
# 2. Category accuracy
# ---------------------------------------------------------

print("=" * 60)
print("CATEGORY ACCURACY (top-5 results per question)")
print("=" * 60)

total_results = 0
correct_results = 0

for question, expected_category in test_questions:
    results = rag.retrieve_products(question, final_k=5, candidate_k=200)

    matches = (results["primary_category"] == expected_category).sum()
    total = len(results)

    total_results += total
    correct_results += matches

    print(f"{matches}/{total}  expected '{expected_category}'  ->  {question}")

category_accuracy = 100 * correct_results / total_results
print(f"\nCategory accuracy: {category_accuracy:.1f}% "
      f"({correct_results}/{total_results} results in expected category)")


# ---------------------------------------------------------
# 3. Filter respect
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FILTER RESPECT (every returned product must obey the filter)")
print("=" * 60)

# Each test is (question, max_price, min_rating, brand).
filter_tests = [
    ("A moisturizer for dry skin", 30.0, None, None),
    ("A long lasting foundation", None, 4.5, None),
    ("A perfume for everyday wear", 50.0, 4.0, None),
    ("A face serum", None, None, "The Ordinary"),
]

filter_checks = 0
filter_passed = 0

for question, max_price, min_rating, brand in filter_tests:
    results = rag.retrieve_products(
        question,
        max_price=max_price,
        min_rating=min_rating,
        brand=brand,
        final_k=5,
        candidate_k=200,
    )

    ok = True

    if max_price is not None:
        ok = ok and (results["price_usd"] <= max_price).all()
    if min_rating is not None:
        ok = ok and (results["rating"] >= min_rating).all()
    if brand is not None:
        ok = ok and (
            results["brand_name"].str.lower() == brand.lower()
        ).all()

    filter_checks += 1
    if ok:
        filter_passed += 1

    print(f"{'PASS' if ok else 'FAIL'}  ({len(results)} results)  "
          f"max_price={max_price} min_rating={min_rating} brand={brand}")

filter_respect = 100 * filter_passed / filter_checks
print(f"\nFilter respect: {filter_respect:.0f}% "
      f"({filter_passed}/{filter_checks} tests fully respected the filters)")
