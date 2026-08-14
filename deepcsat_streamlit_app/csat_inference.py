"""
DeepCSAT inference logic - separated from the Streamlit UI so it can be
loaded and tested independently without
needing a Streamlit runtime.
"""
import re
import string
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def load_artifacts(model_dir='saved_model'):
    from tensorflow import keras
    model = keras.models.load_model(f'{model_dir}/deepcsat_ann_model.keras')
    scaler = joblib.load(f'{model_dir}/scaler.joblib')
    tfidf = joblib.load(f'{model_dir}/tfidf_vectorizer.joblib')
    svd = joblib.load(f'{model_dir}/svd_transformer.joblib')
    feature_columns = joblib.load(f'{model_dir}/feature_columns.joblib')
    medians = joblib.load(f'{model_dir}/medians.joblib')
    outlier_bounds = joblib.load(f'{model_dir}/outlier_bounds.joblib')
    freq_maps = joblib.load(f'{model_dir}/freq_maps.joblib')
    low_card_categories = joblib.load(f'{model_dir}/low_card_categories.joblib')
    low_card_cols = joblib.load(f'{model_dir}/low_card_cols.joblib')
    high_card_cols = joblib.load(f'{model_dir}/high_card_cols.joblib')
    structured_feature_cols = joblib.load(f'{model_dir}/structured_feature_cols.joblib')
    return {
        'model': model, 'scaler': scaler, 'tfidf': tfidf, 'svd': svd,
        'feature_columns': feature_columns, 'medians': medians,
        'outlier_bounds': outlier_bounds, 'freq_maps': freq_maps,
        'low_card_categories': low_card_categories, 'low_card_cols': low_card_cols,
        'high_card_cols': high_card_cols, 'structured_feature_cols': structured_feature_cols,
    }


def load_nltk_tools():
    import nltk
    for res in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    return set(stopwords.words('english')), WordNetLemmatizer()


def clean_text_pipeline(text, stop_words, lemmatizer):
    """Identical preprocessing to the training pipeline in the ML notebook."""
    import contractions
    from nltk.tokenize import word_tokenize

    text = text if isinstance(text, str) else ''
    text = contractions.fix(text) if text else text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text) if text else []
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


def build_feature_row(inputs, art, stop_words, lemmatizer):
    """Turn a single raw interaction (as entered in the UI) into the exact
    same feature vector shape the model was trained on."""

    medians = art['medians']
    bounds = art['outlier_bounds']

    response_time = inputs['response_time_minutes']
    if response_time is None:
        response_time = medians['median_response_time']
        response_time_missing = 1
    else:
        response_time_missing = 0

    item_price = inputs['item_price']
    has_item_price = 1 if item_price is not None else 0
    if item_price is None:
        item_price = medians['median_item_price']

    has_order_info = 1 if inputs['has_order_info'] else 0
    has_remarks = 1 if inputs['customer_remarks'].strip() else 0
    remarks_length = len(inputs['customer_remarks'])

    rt_lo, rt_hi = bounds['response_time_minutes']
    price_lo, price_hi = bounds['Item_price']
    remlen_lo, remlen_hi = bounds['remarks_length']
    response_time = float(np.clip(response_time, rt_lo, rt_hi))
    item_price = float(np.clip(item_price, price_lo, price_hi))
    remarks_length = float(np.clip(remarks_length, remlen_lo, remlen_hi))

    row = {
        'Item_price': item_price,
        'response_time_minutes': response_time,
        'has_remarks': has_remarks,
        'remarks_length': remarks_length,
        'has_order_info': has_order_info,
        'has_item_price': has_item_price,
        'issue_report_hour': inputs['issue_report_hour'],
        'response_time_missing': response_time_missing,
        'response_time_ratio': response_time / (medians['median_response_time_for_ratio'] or 1),
    }

    for col in art['low_card_cols']:
        for cat in art['low_card_categories'][col]:
            row[f'{col}_{cat}'] = 1 if inputs[col] == cat else 0

    for col in art['high_card_cols']:
        freq_map = art['freq_maps'][col]
        key_name = col.lower().replace('-', '_').replace(' ', '_')
        val = inputs[key_name]
        row[col + '_freq'] = freq_map.get(val, 0.0)

    structured_df = pd.DataFrame([row])

    structured_df['response_time_minutes'] = np.log1p(structured_df['response_time_minutes'])
    structured_df['Item_price'] = np.log1p(structured_df['Item_price'].clip(lower=0))
    structured_df['remarks_length'] = np.log1p(structured_df['remarks_length'])

    for col in art['structured_feature_cols']:
        if col not in structured_df.columns:
            structured_df[col] = 0.0
    structured_df = structured_df[art['structured_feature_cols']]

    structured_scaled = art['scaler'].transform(structured_df)
    structured_scaled = pd.DataFrame(structured_scaled, columns=art['structured_feature_cols'])

    cleaned = clean_text_pipeline(inputs['customer_remarks'], stop_words, lemmatizer)
    tfidf_vec = art['tfidf'].transform([cleaned])
    svd_vec = art['svd'].transform(tfidf_vec)
    svd_df = pd.DataFrame(svd_vec, columns=[f'text_svd_{i}' for i in range(svd_vec.shape[1])])

    final_row = pd.concat([structured_scaled.reset_index(drop=True), svd_df.reset_index(drop=True)], axis=1)
    final_row = final_row[art['feature_columns']]
    return final_row


def predict_csat(inputs, art, stop_words, lemmatizer):
    feature_row = build_feature_row(inputs, art, stop_words, lemmatizer)
    probs = art['model'].predict(feature_row, verbose=0)[0]
    pred_class = int(np.argmax(probs)) + 1
    return pred_class, probs
