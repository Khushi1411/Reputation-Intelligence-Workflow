import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import re

st.set_page_config(
    page_title="Reputation Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/final_classified_dataset.csv")
    
    # Create Content column if missing
    if 'Content' not in df.columns:
        df['Content'] = (
            df['Title'].fillna('') + " " + 
            df['Opening Text'].fillna('') + " " + 
            df['Hit Sentence'].fillna('')
        )
        df['Content'] = df['Content'].str.strip()
    
    # Fill nulls
    df['Title'] = df['Title'].fillna('Untitled')
    df['Opening Text'] = df['Opening Text'].fillna('')
    df['Content'] = df['Content'].fillna('')
    df['Source Name'] = df['Source Name'].fillna('Unknown')
    df['Driver'] = df['Driver'].fillna('UNCLASSIFIED')
    df['Sub driver'] = df['Sub driver'].fillna('UNCLASSIFIED')
    df['Sentiment'] = df['Sentiment'].fillna('Neutral')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    return df

df = load_data()

st.sidebar.title("🔍 Filters")

# Driver filter
driver_options = ['All'] + sorted(df['Driver'].unique().tolist())
selected_driver = st.sidebar.selectbox("Reputation Driver", driver_options)

# Sub-driver filter
subdriver_options = ['All'] + sorted(df['Sub driver'].unique().tolist())
selected_subdriver = st.sidebar.selectbox("Sub-Driver", subdriver_options)

# Sentiment filter
sentiment_options = ['All'] + sorted(df['Sentiment'].unique().tolist())
selected_sentiment = st.sidebar.selectbox("Sentiment", sentiment_options)

# Apply filters
filtered_df = df.copy()

if selected_driver != 'All':
    filtered_df = filtered_df[filtered_df['Driver'] == selected_driver]

if selected_subdriver != 'All':
    filtered_df = filtered_df[filtered_df['Sub driver'] == selected_subdriver]

if selected_sentiment != 'All':
    filtered_df = filtered_df[filtered_df['Sentiment'] == selected_sentiment]

st.title("📊 Reputation Intelligence Dashboard")

# Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Mentions", len(filtered_df))

with col2:
    positive = len(filtered_df[filtered_df['Sentiment'] == 'Positive'])
    st.metric("Positive", positive)

with col3:
    neutral = len(filtered_df[filtered_df['Sentiment'] == 'Neutral'])
    st.metric("Neutral", neutral)

with col4:
    negative = len(filtered_df[filtered_df['Sentiment'] == 'Negative'])
    st.metric("Negative", negative)

st.divider()

# Charts
col1, col2 = st.columns(2)

with col1:
    # Sentiment Distribution
    st.subheader("Sentiment Distribution")
    sentiment_counts = filtered_df['Sentiment'].value_counts()
    if len(sentiment_counts) > 0:
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={
                'Positive': '#2ecc71',
                'Neutral': '#f1c40f',
                'Negative': '#e74c3c'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data")

with col2:
    # Driver Distribution
    st.subheader("Driver Distribution")
    driver_counts = filtered_df['Driver'].value_counts()
    if len(driver_counts) > 0:
        fig = px.bar(
            x=driver_counts.index,
            y=driver_counts.values,
            color=driver_counts.index,
            labels={'x': 'Driver', 'y': 'Count'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data")

# Sub-Driver Distribution
st.subheader("Sub-Driver Distribution")
subdriver_counts = filtered_df['Sub driver'].value_counts()
if len(subdriver_counts) > 0:
    fig = px.bar(
        x=subdriver_counts.index,
        y=subdriver_counts.values,
        color=subdriver_counts.index,
        labels={'x': 'Sub-Driver', 'y': 'Count'}
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data")

# Top Discussion Themes
st.subheader("Top Discussion Themes")

# Get top keywords
all_text = ' '.join(filtered_df['Content'].fillna('').astype(str).str.lower())
words = re.findall(r'\b[a-z]{4,}\b', all_text)
stopwords = {'fund', 'mutual', 'icici', 'prudential', 'investment', 'market', 'returns', 
             'equity', 'debt', 'sip', 'nfo', 'portfolio', 'asset', 'bank', 'financial',
             'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her',
             'was', 'one', 'our', 'out', 'has', 'have', 'from', 'with', 'this', 'that'}

keywords = [w for w in words if w not in stopwords and len(w) > 3]
keyword_counts = Counter(keywords).most_common(10)

if keyword_counts:
    top_keywords_df = pd.DataFrame(keyword_counts, columns=['Theme', 'Mentions'])
    fig = px.bar(
        top_keywords_df,
        x='Mentions',
        y='Theme',
        orientation='h',
        labels={'Mentions': 'Count'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No themes available")

st.divider()

st.subheader("Content Explorer")

# Prepare display data
display_df = filtered_df[['Date', 'Source Name', 'Title', 'Opening Text', 'Driver', 'Sub driver', 'Sentiment']].copy()
display_df['Content Preview'] = display_df['Opening Text'].fillna('').str[:150] + '...'

# Show with filters info
st.caption(f"Showing {len(display_df)} records")

# Display table
st.dataframe(
    display_df[['Date', 'Source Name', 'Title', 'Content Preview', 'Driver', 'Sub driver', 'Sentiment']],
    use_container_width=True,
    height=400,
    column_config={
        'Date': st.column_config.DateColumn('Date'),
        'Source Name': st.column_config.TextColumn('Source'),
        'Title': st.column_config.TextColumn('Title'),
        'Content Preview': st.column_config.TextColumn('Content'),
        'Driver': st.column_config.TextColumn('Driver'),
        'Sub driver': st.column_config.TextColumn('Sub-Driver'),
        'Sentiment': st.column_config.TextColumn('Sentiment')
    }
)

st.divider()

st.subheader("Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ Positive Reputation Drivers")
    
    # Calculate positive drivers
    pos_drivers = filtered_df[filtered_df['Sentiment'] == 'Positive']['Driver'].value_counts()
    
    if len(pos_drivers) > 0:
        for driver, count in pos_drivers.items():
            total = len(filtered_df[filtered_df['Driver'] == driver])
            pct = (count / total * 100) if total > 0 else 0
            st.write(f"• **{driver}**: {count} mentions ({pct:.0f}% of driver)")
    else:
        st.info("No positive mentions found")

with col2:
    st.markdown("### ❌ Negative Reputation Drivers")
    
    # Calculate negative drivers
    neg_drivers = filtered_df[filtered_df['Sentiment'] == 'Negative']['Driver'].value_counts()
    
    if len(neg_drivers) > 0:
        for driver, count in neg_drivers.items():
            total = len(filtered_df[filtered_df['Driver'] == driver])
            pct = (count / total * 100) if total > 0 else 0
            st.write(f"• **{driver}**: {count} mentions ({pct:.0f}% of driver)")
    else:
        st.info("No negative mentions found")

# Additional insight - Top Sub-Driver
st.divider()
st.caption("💡 Top Sub-Driver: " + (filtered_df['Sub driver'].value_counts().index[0] if len(filtered_df) > 0 else "N/A"))
st.caption(f"📊 Most mentions from: {filtered_df['Source Name'].value_counts().index[0] if len(filtered_df) > 0 else 'N/A'}")

st.divider()
st.caption(f"Total records: {len(df)} | Filtered: {len(filtered_df)}")