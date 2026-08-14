import numpy as np
from csat_inference import load_artifacts, load_nltk_tools, predict_csat

print("Loading artifacts...")
art = load_artifacts()
stop_words, lemmatizer = load_nltk_tools()
print("Feature vector size:", len(art['feature_columns']))

base = {
    'channel_name': 'Inbound',
    'category': art['low_card_categories']['category'][0],
    'Tenure Bucket': art['low_card_categories']['Tenure Bucket'][0],
    'Agent Shift': art['low_card_categories']['Agent Shift'][0],
    'issue_report_day': 'Monday',
    'issue_report_hour': 11,
    'has_order_info': True,
    'sub_category': 'Not Available',
    'customer_city': 'Not Available',
    'product_category': 'Not Available',
}

# Case 1: fast response, no remarks
case1 = dict(base, response_time_minutes=3.0, item_price=500.0, customer_remarks='')
pred1, probs1 = predict_csat(case1, art, stop_words, lemmatizer)
print(f"\nCase 1 (fast response, no remarks) -> predicted CSAT: {pred1}")
print("  probs:", np.round(probs1, 3))

# Case 2: slow response, angry remarks
case2 = dict(base, response_time_minutes=500.0, item_price=500.0,
             customer_remarks="This is the worst service ever, nobody responded for hours "
                               "and I am extremely disappointed and angry")
pred2, probs2 = predict_csat(case2, art, stop_words, lemmatizer)
print(f"\nCase 2 (slow response, angry remarks) -> predicted CSAT: {pred2}")
print("  probs:", np.round(probs2, 3))

# Case 3: all unknown/missing (leave-1 style inputs, no remarks)
case3 = dict(base, response_time_minutes=None, item_price=None, customer_remarks='')
pred3, probs3 = predict_csat(case3, art, stop_words, lemmatizer)
print(f"\nCase 3 (missing response time & price, no remarks) -> predicted CSAT: {pred3}")
print("  probs:", np.round(probs3, 3))

# Sanity: probabilities should sum to ~1
for name, p in [('case1', probs1), ('case2', probs2), ('case3', probs3)]:
    assert abs(p.sum() - 1.0) < 1e-4, f"{name} probs don't sum to 1: {p.sum()}"

print("\nAll checks passed - no exceptions, probabilities sum to 1.")