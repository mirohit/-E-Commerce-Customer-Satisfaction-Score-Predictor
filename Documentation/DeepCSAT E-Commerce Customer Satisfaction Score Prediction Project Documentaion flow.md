**About This Project:**



Shopzilla is an e-commerce company. Every time a customer contacts support (call, email, or outcall), the customer later gives a CSAT score from 1 to 5 — how happy they were with that interaction. Our job is to build a Deep Learning model (ANN) that can predict this CSAT score just from the details of the interaction (channel used, category of issue, response time, agent info, remarks, etc.) — so the business doesn't have to wait for every customer to fill a survey to know if service was good or bad.





#### **What we have done so far — EDA phase:**



We took the raw data (85,907 support interactions, 20 columns) and did three things:



Checked data quality — shape, duplicates, missing values, datatypes

Cleaned and engineered features — parsed dates, calculated response time, created flags for whether customer left remarks, whether order info was present, etc.

Made 15 charts to understand what affects CSAT, plus a correlation heatmap and pair plot



**Key insights we got:**



* CSAT is very imbalanced: 69% of customers gave a 5. Very few gave 2 or 3. This matters a lot — model can't just be judged on plain accuracy later, because guessing "5" every time would look accurate but be useless.
* Response time matters: The longer it takes to respond to an issue, the lower the CSAT tends to be. This is one of the clearest, most actionable findings.
* If customer writes a remark, satisfaction is usually lower: People mostly write feedback when they're unhappy. So "did customer write something" is itself a warning sign.
* Some categories/products/agent shifts consistently score lower than others — meaning problems are not spread evenly, they're concentrated in specific issue types.



&#x20;   This means CSAT isn't the same everywhere.

&#x20;

&#x20;   For example:



&#x20;   Issue Category A → Average CSAT = 4.6

&#x20;   Issue Category B → Average CSAT = 3.8

&#x20;   Issue Category C → Average CSAT = 3.2



&#x20;   Or perhaps:



&#x20;   Morning shift → Higher CSAT

&#x20;   Night shift   → Lower CSAT



&#x20;   This tells the business:



&#x20;   "There are particular areas where customers are having more problems."



&#x20;   They can investigate those categories, products, or shifts.



* Data quality is patchy: Columns like connected\_handling\_time (99.7% missing), order\_date\_time, Customer\_City, Product\_category, Item\_price (\~80% missing each) — we didn't fake-fill these, we either dropped them or kept the missingness as a signal (flag column).
* No single feature alone predicts CSAT:
This is probably the most important point for understanding why you're using an ANN.



* You checked individual features and found that the CSAT groups overlap.



&#x20;  For example, response time alone doesn't perfectly predict satisfaction:

&#x20;      5-minute response → could get CSAT 2, 3, 4, or 5

&#x20;      And issue category alone doesn't perfectly predict it either.

&#x20;      Same with agent, channel, etc.

&#x20;      But when you combine many features, you may get a better prediction:

&#x20;        Channel + Issue category + Response time + Agent + Remark information + Other features -> ANN -> Predicted CSAT



That's where a neural network can be useful: it can learn combinations and interactions between multiple signals rather than relying on one simple rule.



Data is clean, target is imbalanced, response time + remarks are the strongest early signals, and no single feature is enough alone — so an ANN combining everything makes sense.







#### **What we have done so far — ML ANN Model:**

#### 

* Started with the cleaned data from EDA (85,907 support interactions), added extra engineered features (response time in minutes, remarks length, presence flags for order info/price)



* Proved our hunches with statistics (hypothesis testing) — ran 3 tests to confirm what EDA suggested was actually statistically real, not just a coincidence:

&#x20;        Response time affects CSAT ✅ confirmed

&#x20;        Whether customer wrote a remark affects CSAT ✅ confirmed

&#x20;        Support channel affects CSAT ✅ confirmed



* Prepared the data for the neural network:

&#x20;        Filled missing values sensibly (median values, not random guesses)

&#x20;        Capped extreme outliers (like a ₹5 lakh item price that's clearly a data error)

&#x20;        Converted text categories (channel, category, etc.) into numbers the model can understand

&#x20;        Built a full text processing pipeline for the Customer Remarks column — cleaned the text, removed junk words, reduced words to their root form, then converted it into numbers

&#x20;        using TF-IDF + SVD

&#x20;        Scaled all numbers to the same range



* Handled the imbalance problem — since 69% of customers rated 5/5, we told the model to pay extra attention to the rare cases (1s, 2s, 3s) using class weighting, instead of ignoring them.



* Built and compared 3 different ANN architectures honestly:

&#x20;        Model 1: Simple 2-layer network

&#x20;        Model 2: Deeper network with extra regularization (Dropout, BatchNorm)

&#x20;        Model 3: A more complex tuned version



&#x20;

|Model|Simple Difference|
|-|-|
|Model 1|Basic ANN — 2 hidden layers, no Dropout or Batch Normalization<br />Input → 64 → 32 → Output|
|Model 2|More complex ANN — deeper/wider + Batch Normalization + Dropout<br /><br />Input → 128 → 64 → 32 → Output<br />          ↓      ↓<br />       BatchNorm + Dropout<br /><br />\*\*Batch Normalization:<br />\*\*Batch Normalization (BatchNorm) keeps the values flowing through the neural network in a more stable range while training.<br /><br />This can help the ANN train faster and more smoothly.<br /><br />\*\*Dropout:<br />\*\*Dropout randomly switches off some neurons during training.<br />For example, if you have 100 neurons and dropout is 30%, roughly 30% are temporarily ignored during that training step.<br /><br />\*\*Why?<br />\*\*To stop the model from depending too much on specific neurons and reduce overfitting.|
|Model 3|Tuned/final ANN — moderate architecture + BatchNorm + Dropout + learning-rate adjustment + EarlyStopping<br /><br />Input → 128 → 64 → Output<br />          ↓      ↓<br />       BatchNorm + Dropout<br /><br />**Moderate architecture** = a balanced-sized neural network.<br /><br />In Model 3, ReduceLROnPlateau automatically reduces the learning rate when the model stops improving.<br />**Learning-rate adjustment** = automatically making the learning steps smaller when needed.<br /><br />**EarlyStopping** :<br />EarlyStopping watches the model's validation performance.<br />If the model stops improving for several rounds, it stops training early.<br />Why?<br /><br />To avoid wasting training time and reduce the chance of overfitting.|





* Picked the winner based on real performance, not assumption — measured macro F1 (fair to all classes, not just the majority), and the simplest model (Model 1, tuned) actually won — beat the deeper, fancier ones.



* Saved the winning model with everything needed to reuse it later (this is what powers your Streamlit app).



###### 

###### **What Insights we found ?**

###### 

* CSAT is difficult to predict → Macro F1 is only \~0.25–0.26. The model predicts CSAT 1 and 5 better than the middle scores (2 and 3).
* Simple model performed better → The more complex ANN models did not improve performance. The simpler model was the winner.
* Response time and customer remarks are important → Slower response time and the presence of customer remarks are linked with lower CSAT.
* Limited sentiment understanding → The model does not truly understand the customer's tone. It relies more on features like category and response time than the actual meaning of the customer's text.



#### **What we have done so far — app.py:**

#### 

app.py is the Streamlit web app — it takes everything we built in the ML notebook and turns it into something you can actually use through a browser, without touching code.



* Built a form where you enter details of a customer support interaction: channel, category, sub-category, agent tenure, shift, day, response time, item price, city, product category, and the customer's written remark
* Loaded trained model (and all its preprocessing pieces — scaler, TF-IDF, SVD, etc.) once when the app starts, so it doesn't reload every time you click Predict
* Connected the form to csat\_inference.py — when you click "Predict CSAT Score," it takes whatever you typed, runs it through the exact same cleaning/encoding steps the model was trained on, and feeds it to the ANN
* Displayed the result clearly — the predicted CSAT score (1-5) , a colored warning ("recommend supervisor review" for low scores), and a bar chart showing the model's confidence across all 5 possible scores
* Added an "About this model" tab explaining honestly what the model does well and where it struggles.
* run this command : 1) cd "C:\\Users\\Rohit Sagar Chavan\\Music\\ML Engineering\\DeepCSAT E-Commerce Customer Satisfaction Score Prediction (EDA)\\deepcsat\_streamlit\_app"

&#x20;                     2) python -m streamlit run app.py



#### **What we have done so far — build\_deployment\_artifacts.py:**

#### 

build\_deployment\_artifacts.py is the script that trains your ANN model from scratch and packages everything the app needs to make predictions — it's the bridge between your ML notebook and the Streamlit app.

Think of it like this: your ML notebook is the lab where you experimented with 3 models and figured out which one works best. This script takes that winning model and built it fresh, then packs it into a box (saved\_model/ folder) that the app can grab and use.

#### &#x09;

**What we built into it :**



**1.** Loads and cleans the raw data :

Reads eCommerce\_Customer\_support\_data.csv and applies the exact same cleaning as the EDA/ML notebooks: parses dates, calculates response time, creates flags for missing data, drops the unusable column.





**2.** Prepares the data — and saves the recipe :

Fills missing values with medians, caps extreme outliers, and encodes categories (one-hot for common ones, frequency encoding for rare ones like Sub-category). Crucially, it SAVES the exact medians and bounds it used — so a single new prediction later gets treated the same way as the training data was.



3\. Processes the customer remarks text :

Runs the full text cleaning pipeline on Customer Remarks (lowercase, remove punctuation, remove stopwords, lemmatize), then converts it to numbers using TF-IDF + SVD — and saves those trained tools so new remarks get converted the same way.



4\. Trains the ANN model :

Trains the winning architecture from your ML notebook (Model 1 Tuned: 2 hidden layers, 64→32 units) on this data, using class weighting to handle the imbalance, with early stopping so it doesn't overtrain.



5\. Evaluates it honestly :

Prints Accuracy and Macro F1 on held-out test data — honestly, not hidden — so you can see the model actually works before trusting it (this is where you saw Macro F1 ≈ 0.255 when you ran it yourself).



6\. Saves everything needed for the app:

Saves the trained model AND every preprocessing tool (scaler, TF-IDF, SVD, medians, outlier bounds, frequency maps, category lists) into the saved\_model/ folder — this is what app.py loads every time you launch it.



**Why this file matters specifically ?**



The model/scaler/TF-IDF files your ML notebook originally saved weren't actually complete enough to score a brand-new single customer interaction — they were missing the medians, outlier bounds, and frequency maps I mentioned. This script fixes that gap by saving the complete set of tools needed, which is exactly why your app.py predictions work correctly today.





#### **What we have done so far — csat\_inference.py:**



csat\_inference.py is the prediction engine — it's the code that actually takes one customer interaction (whatever we type into the app) and turns it into a CSAT score. It's not for training — that's what build\_deployment\_artifacts.py was for. This one is purely for using the already-trained model.



Think of it this way: build\_deployment\_artifacts.py built the calculator. csat\_inference.py is the calculator — we hand it numbers, it hands back an answer. app.py is just the buttons and screen we press.



**What we built into it :**



1\. load\_artifacts() — loads everything from saved\_model/:

load\_artifacts() opens the saved\_model/ folder and loads the trained ANN model plus every preprocessing tool: scaler, TF-IDF vectorizer, SVD transformer, medians, outlier bounds, frequency maps, category lists. This only needs to happen once when the app starts.



2\. load\_nltk\_tools() — prepares the text-cleaning tools:

load\_nltk\_tools() prepares the stopword list and the lemmatizer needed to clean text, downloading NLTK resources if they're not already present.





3\. clean\_text\_pipeline() — cleans one remark exactly like training did:

clean\_text\_pipeline() runs a single customer remark through the exact same cleaning steps used in training: expand contractions, lowercase, remove punctuation/URLs/numbers, remove stopwords, tokenize, lemmatize. This has to be IDENTICAL to training, or predictions would be wrong.



4\. build\_feature\_row() — converts inputs into the model's exact input format :

build\_feature\_row() is the most important function — it takes raw form inputs (channel, category, response time, remark text, etc.) and turns them into the exact same 94-number row format the model was trained on: filling missing values with the saved medians, capping outliers with the saved bounds, one-hot/frequency encoding categories, running the text through TF-IDF + SVD, and scaling everything with the saved scaler.



5\. predict\_csat() — the final answer

predict\_csat() ties it together — calls build\_feature\_row(), feeds it to the model, and returns the predicted score (1-5) plus the full probability breakdown across all 5 classes. This is the function app.py calls when we click the Predict button.

