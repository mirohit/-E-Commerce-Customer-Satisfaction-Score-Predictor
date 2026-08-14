"""
DeepCSAT - Deployment Artifact Builder
=======================================
Replicates the exact cleaning / feature engineering / modeling pipeline from
the ML notebook, and saves EVERYTHING needed to score a brand-new, single
customer support interaction at inference time (not just the model itself,
but the medians, outlier bounds, frequency-encoding maps, and one-hot
columns learned during training). This is what the Streamlit app loads.
"""
import numpy as np
import pandas as pd
import re
import string
import os
import time
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings('ignore')

SEED = 42
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---- NLTK resources ----
for res in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4',
            'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.download(res, quiet=True)
    except Exception:
        pass

# ============================================================
# 1. Load + wrangle (identical to EDA / ML notebooks)
# ============================================================
df = pd.read_csv(r"C:\Users\Rohit Sagar Chavan\Music\ML Engineering\DeepCSAT E-Commerce Customer Satisfaction Score Prediction (EDA)\EDA\Input File\eCommerce_Customer_support_data.csv")
df_clean = df.copy()
df_clean.drop_duplicates(inplace=True)

df_clean['Issue_reported at'] = pd.to_datetime(df_clean['Issue_reported at'], format='%d/%m/%Y %H:%M', errors='coerce')
df_clean['issue_responded'] = pd.to_datetime(df_clean['issue_responded'], format='%d/%m/%Y %H:%M', errors='coerce')
df_clean['Survey_response_Date'] = pd.to_datetime(df_clean['Survey_response_Date'], format='%d-%b-%y', errors='coerce')
df_clean['order_date_time'] = pd.to_datetime(df_clean['order_date_time'], format='%d/%m/%Y %H:%M', errors='coerce')

df_clean['response_time_minutes'] = (
    (df_clean['issue_responded'] - df_clean['Issue_reported at']).dt.total_seconds() / 60
)
df_clean.loc[df_clean['response_time_minutes'] < 0, 'response_time_minutes'] = np.nan

df_clean['has_remarks'] = df_clean['Customer Remarks'].notnull().astype(int)
df_clean['remarks_length'] = df_clean['Customer Remarks'].fillna('').apply(len)
df_clean['has_order_info'] = df_clean['Order_id'].notnull().astype(int)
df_clean['has_item_price'] = df_clean['Item_price'].notnull().astype(int)

for col in ['Customer_City', 'Product_category']:
    df_clean[col] = df_clean[col].fillna('Not Available')

df_clean.drop(columns=['connected_handling_time'], inplace=True)
df_clean['issue_report_hour'] = df_clean['Issue_reported at'].dt.hour
df_clean['issue_report_day'] = df_clean['Issue_reported at'].dt.day_name()

# ============================================================
# 2. Missing value imputation (SAVE the medians used)
# ============================================================
df_fe = df_clean.copy()
df_fe['response_time_missing'] = df_fe['response_time_minutes'].isnull().astype(int)

median_response_time = df_fe['response_time_minutes'].median()
median_item_price = df_fe['Item_price'].median()
median_issue_hour = df_fe['issue_report_hour'].median()

df_fe['response_time_minutes'] = df_fe['response_time_minutes'].fillna(median_response_time)
df_fe['Item_price'] = df_fe['Item_price'].fillna(median_item_price)
df_fe['issue_report_hour'] = df_fe['issue_report_hour'].fillna(median_issue_hour)

# ============================================================
# 3. Outlier capping (SAVE the IQR bounds used)
# ============================================================
def iqr_bounds(series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

rt_lower, rt_upper = iqr_bounds(df_fe['response_time_minutes'])
price_lower, price_upper = iqr_bounds(df_fe['Item_price'])
remlen_lower, remlen_upper = iqr_bounds(df_fe['remarks_length'])

df_fe['response_time_minutes'] = df_fe['response_time_minutes'].clip(rt_lower, rt_upper)
df_fe['Item_price'] = df_fe['Item_price'].clip(price_lower, price_upper)
df_fe['remarks_length'] = df_fe['remarks_length'].clip(remlen_lower, remlen_upper)

outlier_bounds = {
    'response_time_minutes': (float(rt_lower), float(rt_upper)),
    'Item_price': (float(price_lower), float(price_upper)),
    'remarks_length': (float(remlen_lower), float(remlen_upper)),
}

# ============================================================
# 4. Categorical encoding (SAVE frequency maps + one-hot categories)
# ============================================================
low_card_cols = ['channel_name', 'category', 'Tenure Bucket', 'Agent Shift', 'issue_report_day']
high_card_cols = ['Sub-category', 'Customer_City', 'Product_category']

low_card_categories = {col: sorted(df_fe[col].dropna().unique().tolist()) for col in low_card_cols}

df_encoded = pd.get_dummies(df_fe, columns=low_card_cols, drop_first=False)

freq_maps = {}
for col in high_card_cols:
    freq_map = df_fe[col].value_counts(normalize=True)
    freq_maps[col] = freq_map.to_dict()
    df_encoded[col + '_freq'] = df_fe[col].map(freq_map)

# ============================================================
# 5. Text pipeline
# ============================================================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text_pipeline(text):
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

final_text = df_fe['Customer Remarks'].fillna('').apply(clean_text_pipeline)

tfidf = TfidfVectorizer(max_features=300, min_df=5, ngram_range=(1, 2))
text_tfidf = tfidf.fit_transform(final_text)

svd = TruncatedSVD(n_components=50, random_state=SEED)
text_svd = svd.fit_transform(text_tfidf)
text_svd_df = pd.DataFrame(text_svd, columns=[f'text_svd_{i}' for i in range(50)], index=df_fe.index)

# ============================================================
# 6. Feature manipulation + selection
# ============================================================
drop_cols = [
    'Unique id', 'Order_id', 'Customer Remarks', 'Agent_name', 'Supervisor', 'Manager',
    'Sub-category', 'Customer_City', 'Product_category',
    'order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date'
]
df_model = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])

median_rt = df_fe['response_time_minutes'].median()
df_model['response_time_ratio'] = df_fe['response_time_minutes'] / (median_rt if median_rt else 1)

target_col = 'CSAT Score'
structured_feature_cols = [c for c in df_model.columns if c != target_col]
X_structured = df_model[structured_feature_cols].astype(float)
y = df_model[target_col].astype(int)

# ============================================================
# 7. Transform + scale (SAVE the scaler)
# ============================================================
X_structured['response_time_minutes'] = np.log1p(X_structured['response_time_minutes'])
X_structured['Item_price'] = np.log1p(X_structured['Item_price'].clip(lower=0))
X_structured['remarks_length'] = np.log1p(X_structured['remarks_length'])

scaler = StandardScaler()
X_structured_scaled = pd.DataFrame(
    scaler.fit_transform(X_structured), columns=X_structured.columns, index=X_structured.index
)

X_final = pd.concat([X_structured_scaled.reset_index(drop=True), text_svd_df.reset_index(drop=True)], axis=1)
y_final = y.reset_index(drop=True) - 1

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=SEED, stratify=y_final
)

class_weights_arr = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: w for cls, w in zip(np.unique(y_train), class_weights_arr)}

# ============================================================
# 8. Train the winning architecture (Model 1 Tuned: 64/32 units, lr=0.001)
# ============================================================
n_classes = 5
input_dim = X_train.shape[1]

final_model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])
final_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy', metrics=['accuracy'])
es = callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
final_model.fit(X_train, y_train, validation_split=0.15, epochs=15, batch_size=512,
                 class_weight=class_weight_dict, callbacks=[es], verbose=0)

y_pred = np.argmax(final_model.predict(X_test, verbose=0), axis=1)
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
print(f"Deployment model retrained -> Accuracy: {acc:.4f}, Macro F1: {f1_macro:.4f}")
print(classification_report(y_test, y_pred, target_names=[f'CSAT {i+1}' for i in range(5)]))

# ============================================================
# 9. Save EVERYTHING needed for single-row inference
# ============================================================
os.makedirs('saved_model', exist_ok=True)

final_model.save('saved_model/deepcsat_ann_model.keras')
joblib.dump(scaler, 'saved_model/scaler.joblib')
joblib.dump(tfidf, 'saved_model/tfidf_vectorizer.joblib')
joblib.dump(svd, 'saved_model/svd_transformer.joblib')
joblib.dump(list(X_final.columns), 'saved_model/feature_columns.joblib')

joblib.dump({
    'median_response_time': float(median_response_time),
    'median_item_price': float(median_item_price),
    'median_issue_hour': float(median_issue_hour),
    'median_response_time_for_ratio': float(median_rt),
}, 'saved_model/medians.joblib')

joblib.dump(outlier_bounds, 'saved_model/outlier_bounds.joblib')
joblib.dump(freq_maps, 'saved_model/freq_maps.joblib')
joblib.dump(low_card_categories, 'saved_model/low_card_categories.joblib')
joblib.dump(low_card_cols, 'saved_model/low_card_cols.joblib')
joblib.dump(high_card_cols, 'saved_model/high_card_cols.joblib')
joblib.dump(structured_feature_cols, 'saved_model/structured_feature_cols.joblib')

print("\nAll deployment artifacts saved to 'saved_model/':")
print(sorted(os.listdir('saved_model')))
