import streamlit as st
import pandas as pd
import plotly.express as px
from snowflake.cortex import complete
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session

# Create Snowflake connection
@st.cache_resource
def get_snowflake_session():
    connection_parameters = {
        "account": st.secrets["connections"]["snowflake"]["account"],
        "user": st.secrets["connections"]["snowflake"]["user"],
        "password": st.secrets["connections"]["snowflake"]["password"],
        "role": st.secrets["connections"]["snowflake"]["role"],
        "warehouse": st.secrets["connections"]["snowflake"]["warehouse"],
        "database": st.secrets["connections"]["snowflake"]["database"],
        "schema": st.secrets["connections"]["snowflake"]["schema"]
    }
    return Session.builder.configs(connection_parameters).create()

session = get_snowflake_session()

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

# --- Sentiment Score by Carrier and Region ---
st.subheader("Average Sentiment Score by Carrier and Region")

carrier_region_sentiment = (
    filtered_df.groupby(['CARRIER', 'REGION'])['SENTIMENT_SCORE']
    .mean()
    .reset_index()
)

fig2 = px.bar(
    carrier_region_sentiment,
    x="CARRIER",
    y="SENTIMENT_SCORE",
    color="REGION",
    barmode="group",
    title="Average Sentiment Score by Carrier and Region",
    labels={"SENTIMENT_SCORE": "Mean Sentiment Score", "CARRIER": "Carrier", "REGION": "Region"},
)
fig2.update_layout(
    xaxis_title="Carrier", 
    yaxis_title="Mean Sentiment Score",
    legend_title="Region"
)
st.plotly_chart(fig2, use_container_width=True)

# --- Carrier Performance vs Others ---
st.subheader("Carrier Sentiment Score vs Others")

# Calculate mean sentiment for each carrier
carrier_means = filtered_df.groupby('CARRIER')['SENTIMENT_SCORE'].mean()

# Calculate the difference for each carrier vs all others
carrier_diff = []
for carrier in carrier_means.index:
    carrier_score = carrier_means[carrier]
    others_score = carrier_means[carrier_means.index != carrier].mean()
    diff = carrier_score - others_score
    carrier_diff.append({
        'CARRIER': carrier,
        'DIFFERENCE': diff,
        'CARRIER_SCORE': carrier_score,
        'OTHERS_SCORE': others_score
    })

diff_df = pd.DataFrame(carrier_diff).sort_values('DIFFERENCE', ascending=True)

fig3 = px.bar(
    diff_df,
    x="CARRIER",
    y="DIFFERENCE",
    title="Carrier Sentiment Difference vs All Others",
    labels={"DIFFERENCE": "Difference from Others", "CARRIER": "Carrier"},
    color="DIFFERENCE",
    color_continuous_scale=["red", "yellow", "green"]
)
fig3.update_layout(
    xaxis_title="Carrier",
    yaxis_title="Sentiment Difference (vs Others Mean)",
    yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
)
st.plotly_chart(fig3, use_container_width=True)

# --- Carrier Performance vs Others by Region ---
st.subheader("Carrier Sentiment Score vs Others by Region")

# Calculate mean sentiment for each carrier and region
carrier_region_means = filtered_df.groupby(['CARRIER', 'REGION'])['SENTIMENT_SCORE'].mean()

# Calculate the difference for each carrier vs all others within each region
carrier_region_diff = []
for region in filtered_df['REGION'].unique():
    region_data = carrier_region_means.xs(region, level='REGION')
    
    for carrier in region_data.index:
        carrier_score = region_data[carrier]
        others_score = region_data[region_data.index != carrier].mean()
        diff = carrier_score - others_score
        carrier_region_diff.append({
            'CARRIER': carrier,
            'REGION': region,
            'DIFFERENCE': diff,
            'CARRIER_SCORE': carrier_score,
            'OTHERS_SCORE': others_score
        })

diff_df = pd.DataFrame(carrier_region_diff)

fig3 = px.bar(
    diff_df,
    x="CARRIER",
    y="DIFFERENCE",
    color="REGION",
    barmode="group",
    title="Carrier Sentiment Difference vs Others by Region",
    labels={"DIFFERENCE": "Difference from Others", "CARRIER": "Carrier", "REGION": "Region"},
)
fig3.update_layout(
    xaxis_title="Carrier",
    yaxis_title="Sentiment Difference (vs Others Mean in Region)",
    yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
    legend_title="Region"
)
st.plotly_chart(fig3, use_container_width=True)

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
