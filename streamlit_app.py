import streamlit as st
import pandas as pd
import plotly.express as px
from snowflake.cortex import complete
from snowflake.snowpark.context import get_active_session

# Get the active Snowflake session
session = get_active_session()

# Load data
df = session.table("reviews_sentiment_big").to_pandas()

# App UI
st.title("Avalanche Streamlit App ❄️")

# Sidebar filters
carriers = df['CARRIER'].unique()
selected_carriers = st.sidebar.multiselect(
    "Select Carriers:",
    options=carriers,
    default=carriers
)
filtered_df = df[df['CARRIER'].isin(selected_carriers)]

# Data preview
st.subheader("Data Sample")
st.dataframe(filtered_df.sample(5))

# --- Visualization: Average Sentiment by Region ---
st.subheader("Average Sentiment by Region")

region_sentiment = (
    filtered_df.groupby("REGION")['SENTIMENT_SCORE']
    .mean()
    .reset_index()
    .sort_values("SENTIMENT_SCORE", ascending=True)
)

fig = px.bar(
    region_sentiment,
    x="SENTIMENT_SCORE",
    y="REGION",
    orientation="h",
    title="Average Sentiment by Region",
    labels={"SENTIMENT_SCORE": "Sentiment Score", "REGION": "Region"},
)
fig.update_layout(xaxis_title="Sentiment Score", yaxis_title="Region")
st.plotly_chart(fig, use_container_width=True)

# --- Delivery Issues by Region and Carrier ---
st.subheader("Delivery Issues by Region and Carrier")

grouped_issues = (
    filtered_df.groupby(['REGION', 'CARRIER'])['SENTIMENT_SCORE']
    .mean()
    .reset_index()
)
st.dataframe(grouped_issues)

# --- Chatbot Assistant ---
st.subheader("Ask Questions About the Data")

user_question = st.text_input("Enter your question here:")

if user_question:
    # Convert the DataFrame to string context for the LLM
    df_string = df.to_string(index=False)
    response = complete(
        model="claude-3-5-sonnet",
        prompt=f"Answer this question using the dataset: {user_question} <context>{df_string}</context>",
        session=session
    )
    st.write(response)
