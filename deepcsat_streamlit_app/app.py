"""
DeepCSAT - E-Commerce Customer Satisfaction Score Prediction
Local Streamlit deployment app.

Run locally with:
    python -m streamlit run app.py

Requires the 'saved_model/' folder (produced by build_deployment_artifacts.py)
and csat_inference.py to be in the same directory as this script.
"""
import warnings

import streamlit as st
import matplotlib.pyplot as plt

from csat_inference import load_artifacts, load_nltk_tools, predict_csat

warnings.filterwarnings('ignore')

st.set_page_config(page_title="DeepCSAT - CSAT Score Predictor", page_icon="", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_resource
def get_nltk_tools():
    return load_nltk_tools()


st.title("DeepCSAT: E-Commerce Customer Satisfaction Score Predictor")
st.caption("Local deployment of the ANN model built in the DeepCSAT capstone project (Shopzilla support data)")

with st.spinner("Loading model and preprocessing pipeline..."):
    art = get_artifacts()
    stop_words, lemmatizer = get_nltk_tools()

st.success("Model loaded and ready.")

tab1, tab2 = st.tabs([" Predict CSAT Score", " About this model"])

with tab1:
    st.subheader("Enter interaction details")
    col1, col2 = st.columns(2)

    with col1:
        channel_name = st.selectbox("Channel", art['low_card_categories']['channel_name'])
        category = st.selectbox("Category", art['low_card_categories']['category'])
        sub_category = st.text_input("Sub-category", value="Product Specific Information")
        tenure_bucket = st.selectbox("Agent Tenure Bucket", art['low_card_categories']['Tenure Bucket'])
        agent_shift = st.selectbox("Agent Shift", art['low_card_categories']['Agent Shift'])
        issue_report_day = st.selectbox("Day Issue Reported", art['low_card_categories']['issue_report_day'])

    with col2:
        issue_report_hour = st.slider("Hour Issue Reported (0-23)", 0, 23, 12)
        response_time_minutes = st.number_input(
            "Response Time (minutes) - leave at -1 if unknown", min_value=-1.0, value=15.0, step=1.0
        )
        has_order_info = st.checkbox("Order info available for this interaction?", value=True)
        item_price = st.number_input(
            "Item Price (leave at -1 if unknown/not applicable)", min_value=-1.0, value=500.0, step=10.0
        )
        customer_city = st.text_input("Customer City (or leave blank)", value="")
        product_category = st.text_input("Product Category (or leave blank)", value="")

    customer_remarks = st.text_area(
        "Customer Remarks (written feedback, if any)",
        value="",
        placeholder="e.g. 'The agent was very slow to respond and did not resolve my issue'"
    )

    if st.button("Predict CSAT Score", type="primary"):
        inputs = {
            'channel_name': channel_name,
            'category': category,
            'Tenure Bucket': tenure_bucket,
            'Agent Shift': agent_shift,
            'issue_report_day': issue_report_day,
            'issue_report_hour': issue_report_hour,
            'response_time_minutes': None if response_time_minutes < 0 else response_time_minutes,
            'has_order_info': has_order_info,
            'item_price': None if item_price < 0 else item_price,
            'sub_category': sub_category if sub_category else 'Not Available',
            'customer_city': customer_city if customer_city else 'Not Available',
            'product_category': product_category if product_category else 'Not Available',
            'customer_remarks': customer_remarks,
        }

        pred_class, probs = predict_csat(inputs, art, stop_words, lemmatizer)

        st.divider()
        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            emoji = {1: "angry", 2: "sad", 3: "neutral", 4: "happy", 5: "very happy"}[pred_class]
            st.metric("Predicted CSAT Score", f"{pred_class} / 5 {emoji}")
            if pred_class <= 2:
                st.error("High risk of dissatisfaction — recommend supervisor review.")
            elif pred_class == 3:
                st.warning("Neutral/uncertain — worth a quick quality check.")
            else:
                st.success("Likely a satisfied customer.")

        with result_col2:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar([1, 2, 3, 4, 5], probs, color=['#d62728', '#ff7f0e', '#bcbd22', '#98df8a', '#2ca02c'])
            ax.set_xlabel("CSAT Score")
            ax.set_ylabel("Predicted Probability")
            ax.set_title("Predicted probability by CSAT class")
            ax.set_xticks([1, 2, 3, 4, 5])
            st.pyplot(fig)

with tab2:
    st.markdown("""
    ### About this model
    This app deploys the **final selected ANN model** from the DeepCSAT capstone project:
    a 2-hidden-layer network (64 → 32 units, ReLU, Adam optimizer at lr=0.001), chosen
    because it achieved the best **macro F1 score** among all 5 model/configuration
    combinations tested in the ML notebook — including deeper, more heavily regularized
    architectures that did *not* outperform it.

    **Pipeline applied to every prediction** (identical to training):
    - Missing value imputation (median-based, with presence flags preserved)
    - IQR-based outlier capping on response time, item price, and remarks length
    - One-hot encoding (channel, category, tenure, shift, day) + frequency encoding
      (sub-category, city, product category)
    - Full NLP pipeline on customer remarks: contraction expansion → lowercasing →
      punctuation/URL/digit removal → stopword removal → tokenization → lemmatization
      → TF-IDF (300 features) → Truncated SVD (50 components)
    - Log-transform + StandardScaler on skewed numeric features

    **Honest limitation:** overall macro F1 is modest (~0.25-0.26) — the model is
    noticeably better at spotting clearly happy (CSAT 5) or clearly upset (CSAT 1)
    customers than the ambiguous middle (CSAT 2/3), which is a real, reported
    limitation of the available data rather than something hidden from this app.
    Treat predictions as a **triage signal for supervisor review**, not a
    ground-truth substitute for the actual customer survey.
    """)

st.divider()
st.caption("DeepCSAT Capstone Project · Local deployment via Streamlit · Model: ANN (Keras/TensorFlow)")
