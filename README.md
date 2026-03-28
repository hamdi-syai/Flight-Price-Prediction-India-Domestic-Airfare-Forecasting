# ✈️ Flight Price Prediction: India Domestic Airfare Forecasting

## 📌 Overview
This project builds a machine learning regression model to predict domestic flight ticket prices in India based on flight attributes such as airline, route, class, number of stops, departure time, and days left before departure. The goal is to help consumers and travel platforms make smarter booking decisions with more accurate price estimations.

## 🎯 Objectives
- Build an accurate flight price prediction model
- Identify the key factors that influence ticket pricing
- Provide optimal booking time recommendations
- Support travel platforms with reliable price estimation tools

## 📂 Dataset
The dataset contains India domestic flight records with the following attributes:

| Feature | Type | Description |
|---|---|---|
| `airline` | Categorical | Airline name (SpiceJet, AirAsia, Vistara, etc.) |
| `flight` | Categorical | Flight code identifier (dropped — not informative) |
| `source_city` | Categorical | City of departure |
| `departure_time` | Categorical | Time-of-day period of departure |
| `stops` | Categorical | Number of stops (zero, one, two or more) |
| `arrival_time` | Categorical | Time-of-day period of arrival |
| `destination_city` | Categorical | City of arrival |
| `class` | Categorical | Cabin class (Economy / Business) |
| `duration` | Numerical | Flight duration in hours |
| `days_left` | Numerical | Days remaining before departure |
| `price` | Numerical | Ticket price in INR *(target variable)* |

> **Source:** India Domestic Flight Price Dataset — Kaggle | **Records:** 300,000+

## 🛠️ Tools & Technologies
- Python
- Pandas & NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Google Colab

## 🔍 Key Analysis Steps

### 1. Data Exploration
- Reviewed dataset structure, data types, and basic statistics
- Confirmed no missing values and no duplicate rows
- Identified unique values across all categorical features
- Analyzed price distribution per airline, class, stops, route, and departure time

### 2. Exploratory Data Analysis (EDA)
- Distribution of target variable (`price`) — histogram & boxplot
- Categorical features vs. price (bar charts per group)
- Numerical features vs. price (scatter plots, correlation heatmap)
- Price trends by booking window (`days_left`) for Economy and Business separately
- Route-level average price ranking (horizontal bar chart)
- Airline × Class pivot heatmap

### 3. Data Preprocessing
- Dropped `flight` column (ID column with no predictive value)
- Handled outliers **per flight class** using IQR method (multiplier = 3.0) to avoid cross-class distortion
- Applied ordinal encoding: `stops`, `departure_time`, `arrival_time`
- Applied binary encoding: `class` (Economy=0, Business=1)
- Applied label encoding: `airline`, `source_city`, `destination_city`

### 4. Feature Engineering
- `is_offpeak` — flag for off-peak departure time slots
- `urgency` — urgency level derived from `days_left` remaining
- `duration_cat` — categorical bucket of flight duration
- Correlation analysis post-encoding to validate feature relevance

### 5. Modelling — Baseline
Trained 5 regression models on an 80/20 train-test split:

| Model | Notes |
|---|---|
| Linear Regression | Simple linear baseline |
| Ridge Regression | Regularized linear model (α=1.0) |
| Decision Tree | Tree-based, non-linear |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosted trees |

### 6. Hyperparameter Tuning
Applied `RandomizedSearchCV` (cv=3) on the two top-performing baseline models:
- **Random Forest** — tuned: n_estimators, max_depth, min_samples_split, max_features, bootstrap
- **Gradient Boosting** — tuned: n_estimators, learning_rate, max_depth, subsample

### 7. Model Evaluation
Metrics used:

| Metric | Description |
|---|---|
| R² Score | Proportion of variance explained by the model |
| MAE | Mean Absolute Error (INR) |
| RMSE | Root Mean Squared Error (INR) |
| MAPE | Mean Absolute Percentage Error (%) |

5-fold cross-validation performed on the best model to confirm stability and generalization.

### 8. Business Use Case Evaluation
- Evaluated model performance **per flight class** (Economy vs. Business)
- Analyzed average pricing by booking window buckets (1–7 days up to 36–49 days)
- Generated actionable consumer recommendations based on model findings

## 📊 Key Insights
- **Flight class** is the strongest price driver — Business tickets are significantly more expensive than Economy on the same route
- **Days left before departure** shows a clear inverse trend — prices spike sharply within 7 days of departure
- **Number of stops** correlates positively with price, particularly in Economy class
- **Vistara and Air India** consistently price higher than budget carriers (SpiceJet, AirAsia) across all routes
- **Early morning and late night** departures tend to be cheaper than peak-hour slots
- The optimal booking window for Economy class is **22–35 days** before departure

## 📈 Business Recommendations
- Book Economy tickets **3–5 weeks in advance** to avoid last-minute price surges
- Consider **off-peak departure slots** (early morning / late night) for lower fares on the same route
- Travel platforms can integrate this model to display a "Fair Price" indicator on listings
- Use predicted price as a baseline to alert consumers when current prices are above expected range
- Airlines can leverage the model inversely for dynamic pricing and yield management strategies

## 📸 Visualizations
<img width="1784" height="1011" alt="téléchargement (2)" src="https://github.com/user-attachments/assets/bbe40998-4177-4b9c-af44-8ebf561cade3" />
<img width="1784" height="492" alt="téléchargement (3)" src="https://github.com/user-attachments/assets/e51890ba-b126-418f-aeaf-9c1315cb37b8" />


## 🚀 Future Improvements
- Incorporate real-time flight data via airline APIs for live price monitoring
- Add seat availability as a scarcity-based pricing signal
- Build a Streamlit web app for interactive price prediction by users
- Experiment with XGBoost and LightGBM for further R² improvements
- Deploy the best model as a REST API for integration with travel platforms

## 👤 Author
**Syaibatul Hamdi** — Data Consultant based in France

---
⭐ Feel free to explore and give feedback!
