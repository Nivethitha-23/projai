import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
import io
import subprocess
import sys

# --- Code to handle library installation if not present ---
# This ensures the app can run even if the user hasn't installed all packages
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.warning("`plotly` library not found. Installing now...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
        import plotly.express as px
        import plotly.graph_objects as go
        st.success("`plotly` installed successfully! Please re-run the app.")
    except Exception as e:
        st.error(f"Failed to install `plotly`. Please install it manually: `pip install plotly`")
        st.stop()


# --- Ensure nltk resources are downloaded ---
# VADER lexicon is needed for sentiment analysis
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except nltk.downloader.DownloadError:
    nltk.download("vader_lexicon")

# --- Streamlit Page Configuration ---
# Sets up a wide layout and a professional title for the page
st.set_page_config(layout="wide", page_title="Professional Competitor Tracker", initial_sidebar_state="expanded")

# -----------------------------
# 1️⃣ Load Data & Feature Engineering (Cached for efficiency)
# -----------------------------
@st.cache_data
def load_data(path="realistic_products_dataset.csv"):
    """
    Loads and caches the dataset. 
    @st.cache_data decorator ensures this function runs only once,
    improving app performance.
    """
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        df['customer_review'] = df['customer_review'].astype(str)
        return df
    except FileNotFoundError:
        st.error(f"Error: Dataset file not found at '{path}'. Please ensure the file is in the correct directory.")
        st.stop()
        return pd.DataFrame()

df_raw = load_data()

@st.cache_data
def preprocess_data(df_raw):
    """
    Performs all necessary data preprocessing and feature engineering.
    This includes sentiment analysis and creating lagged/rolling features.
    """
    df = df_raw.copy()
    sia = SentimentIntensityAnalyzer()
    
    # Apply sentiment analysis to create a compound score and a label
    df["sentiment_score"] = df["customer_review"].apply(lambda x: sia.polarity_scores(x)["compound"])
    df["sentiment_label"] = df["sentiment_score"].apply(
        lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
    )
    
    # Create a binary flag for promotions based on discount
    df["promo_flag"] = df["discount"].apply(lambda x: 1 if x > 0 else 0)
    
    # Sort data for time-series feature creation
    df = df.sort_values(["product_name", "date"])
    
    # Create lagged features for price and promo flag
    df["price_lag_1"] = df.groupby("product_name")["price"].shift(1)
    
    # Create a 3-period rolling average of the price
    df["price_roll_3"] = df.groupby("product_name")["price"].shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    df["promo_lag_1"] = df.groupby("product_name")["promo_flag"].shift(1)
    
    # Drop rows with NaN values resulting from feature engineering
    df_feat = df.dropna().reset_index(drop=True)
    return df, df_feat

df_full, df_feat = preprocess_data(df_raw)

# -----------------------------
# 2️⃣ Sidebar for Controls
# -----------------------------
# A professional sidebar for user interaction
st.sidebar.title("Competitor Tracker 📈")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Product Selection")

product_select = st.sidebar.selectbox(
    "Choose a Product to Analyze", 
    df_full["product_name"].unique()
)

# -----------------------------
# 3️⃣ Main Dashboard with Plotly Visualizations
# -----------------------------
# Main title and subtitle of the dashboard
st.title("🛒 Real-Time Competitor Strategy Tracker")
st.markdown("###  Predictive Modeling ")
st.markdown("A dashboard to help you analyze market trends and forecast competitor strategies.")

# -----------------------------
# Future Price & Promo Forecast
# -----------------------------
st.header("📈 Future Pricing & Promo Forecast")
st.markdown(f"#### Analyzing **{product_select}**")

# Split data and train models (cached for efficiency)
@st.cache_resource
def train_models(X_price_data, y_price_data, X_promo_data, y_promo_data):
    """
    Trains and caches the price and promo prediction models.
    @st.cache_resource ensures the models are trained only once.
    """
    if X_price_data.empty or X_promo_data.empty:
        return None, None
        
    # Price Prediction Model using LightGBM
    X_train_price, _, y_train_price, _ = train_test_split(
        X_price_data, y_price_data, test_size=0.2, random_state=42, shuffle=False
    )
    train_data_price = lgb.Dataset(X_train_price, label=y_train_price)
    params_price = {"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42}
    model_price = lgb.train(params_price, train_data_price, num_boost_round=200)

    # Promo Prediction Model using LightGBM (binary classification)
    X_train_promo, _, y_train_promo, _ = train_test_split(
        X_promo_data, y_promo_data, test_size=0.2, random_state=42, shuffle=False
    )
    train_data_promo = lgb.Dataset(X_train_promo, label=y_train_promo)
    params_promo = {"objective": "binary", "metric": "auc", "verbosity": -1, "seed": 42}
    model_promo = lgb.train(params_promo, train_data_promo, num_boost_round=200)

    return model_price, model_promo

# Train models on full dataset to make predictions on all products
features = ["price_lag_1", "price_roll_3", "promo_lag_1"]
model_price_all, model_promo_all = train_models(
    df_feat[features], df_feat["price"], df_feat[features], df_feat["promo_flag"]
)

# Get data for the selected product
df_product = df_feat[df_feat["product_name"] == product_select].copy()

if not df_product.empty and model_price_all and model_promo_all:
    col_price, col_promo = st.columns(2)
    
    # Predict future price based on the last data point
    last_data_point = df_product.iloc[-1][features].values.reshape(1, -1)
    future_price_pred = model_price_all.predict(last_data_point)[0]
    
    with col_price:
        st.subheader("Price Forecast")
        st.metric("Predicted Next Price", f"₹{future_price_pred:,.2f}")
        
        # Plotly chart for price trend - Note: This chart has built-in zoom, pan, and hover
        price_plot_df = df_product[['date', 'price']].copy()
        
        # Create a Plotly figure object
        fig_price = go.Figure()
        
        # Add historical price line
        fig_price.add_trace(go.Scatter(
            x=price_plot_df['date'],
            y=price_plot_df['price'],
            mode='lines+markers',
            name='Historical Price'
        ))
        
        # Add a marker for the future prediction point
        fig_price.add_trace(go.Scatter(
            x=[price_plot_df['date'].max() + pd.Timedelta(days=1)],
            y=[future_price_pred],
            mode='markers',
            marker=dict(size=12, color='red'),
            name='Predicted Future Price'
        ))
        
        # Update the layout for a clean look
        fig_price.update_layout(
            title=f"Price Trend for {product_select}",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_price, use_container_width=True)

    # Predict future promo probability based on the last data point
    promo_prob_pred = model_promo_all.predict(last_data_point)[0]
    with col_promo:
        st.subheader("Promotion Forecast")
        st.metric("Probability of Promotion", f"{promo_prob_pred * 100:.2f}%")
        
        # Display a status based on probability
        if promo_prob_pred > 0.5:
            st.warning("High probability of a future promotion. Prepare a counter-strategy.")
        else:
            st.info("Low probability of a promotion. Current strategy is likely stable.")
            
# -----------------------------
# Customer Satisfaction Analysis
# -----------------------------
st.header("📊 Customer Satisfaction & Sentiment Analysis")

# Get sentiment data for the selected product
df_sent = df_full[df_full["product_name"] == product_select]
sent_count = df_sent["sentiment_label"].value_counts()

# Pie chart for sentiment distribution
fig_pie = px.pie(
    names=sent_count.index, 
    values=sent_count.values, 
    title=f"Sentiment Distribution for {product_select}",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
st.plotly_chart(fig_pie, use_container_width=True)

# --- Conditional conclusion based on average sentiment ---
avg_sentiment = df_sent["sentiment_score"].mean()
if avg_sentiment > 0.05:
    st.success(f"✅ Customers generally feel **positive** about {product_select}.")
elif avg_sentiment < -0.05:
    st.error(f"⚠️ Customers generally feel **negative** about {product_select}.")
else:
    st.warning(f"😐 Customers are **neutral** about {product_select}.")

# Bar chart for average rating per product
st.markdown("---")
st.subheader("Average Customer Rating by Product")
product_ratings = df_full.groupby("product_name")["rating"].mean().reset_index()
fig_bar = px.bar(
    product_ratings, 
    x="product_name", 
    y="rating", 
    title="Average Rating Across All Products",
    color="rating",
    color_continuous_scale=px.colors.sequential.Viridis
)
fig_bar.update_layout(xaxis_title="Product", yaxis_title="Average Rating")
st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# New Tab for Product Insights
# -----------------------------
st.markdown("---")
st.header("Product Insights")
st.markdown("### Price vs. Sentiment ")

# Group data by product to get average metrics for the scatter plot
product_summary = df_full.groupby('product_name').agg(
    avg_price=('price', 'mean'),
    avg_sentiment=('sentiment_score', 'mean'),
    review_count=('review_count', 'sum')
).reset_index()

fig_scatter = px.scatter(
    product_summary,
    x='avg_price',
    y='avg_sentiment',
    size='review_count',
    hover_name='product_name',
    title='Price vs. Sentiment by Product',
    labels={
        'avg_price': 'Average Price (₹)',
        'avg_sentiment': 'Average Sentiment Score',
        'review_count': 'Number of Reviews'
    },
    template='plotly_dark'
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("### Analysis & Conclusion")
st.info("""
This scatter plot reveals the relationship between a product's price and its customer sentiment. Products in the top-right quadrant are high-priced with high sentiment, indicating a successful premium strategy. Products in the bottom-right may be overpriced, while those in the top-left could be considered 'value champions.' The size of each bubble corresponds to the number of reviews, highlighting the most discussed products.
""")

# -----------------------------
# Raw Data Expander
# -----------------------------
st.markdown("---")
with st.expander("🔍 View Raw Dataset"):
    st.dataframe(df_raw)
