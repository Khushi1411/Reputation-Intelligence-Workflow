import pandas as pd
import re

df = pd.read_excel("data/Dataset.xlsx")

#Dataset Exploration
print(df.head())
print(df.columns)
print(df.shape)
print(df.info())
print(df.isnull().sum())


#Creating content column combining title, opening text and hit sentence
df["Content"] = (
    df["Title"].fillna("") + " " +
    df["Opening Text"].fillna("") + " " +
    df["Hit Sentence"].fillna("")
)

df["Content"] = df["Content"].str.strip()

#Removing empty content records
before_empty = len(df)
df = df[df["Content"] != ""]
after_empty = len(df)
print(f"\nEmpty Content Records Removed: {before_empty - after_empty}")

#REMOVAL OF IRRELEVANT RECORDS
relevant_keywords = [
    # Brand mentions
    'icici prudential', 'icici pru', 'icici prudential amc',
    
    # Mutual fund terms
    'mutual fund', 'sip', 'nfo', 'new fund offer', 'etf', 
    'index fund', 'hybrid fund', 'equity fund', 'debt fund',
    'arbitrage fund', 'fund manager',
    
    # Investment terms
    'investment', 'invest', 'portfolio', 'returns', 'cagr',
    'aum', 'assets under management', 'dividend',
    
    # Financial terms
    'equity', 'debt', 'bond', 'gold', 'silver', 
    'bank', 'banking', 'nifty', 'sensex', 'market',
    
    # Product types
    'large cap', 'mid cap', 'small cap', 'flexi cap', 
    'multi cap', 'business cycle', 'retirement fund',
    'children fund', 'value fund', 'bluechip fund',
    
    # Digital experience
    'app', 'mobile app', 'digital experience', 
    'registration', 'transaction', 'investment platform',
    
    # CSR and compliance
    'csr', 'corporate social responsibility', 'governance',
    'compliance', 'sebi', 'regulatory'
]

def is_relevant(text):
    """Check if text contains any relevant keyword."""
    if pd.isna(text) or text == "":
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in relevant_keywords)

# Apply irrelevant record removal
before_irrelevant = len(df)
df = df[df["Content"].apply(is_relevant)]
after_irrelevant = len(df)
print(f"Irrelevant Records Removed: {before_irrelevant - after_irrelevant}")

#Standardize Sentiment
df["Sentiment"] = (
    df["Sentiment"]
    .astype(str)
    .str.strip()
    .str.title()
)

#Standardize Source Name
df["Source Name"] = (
    df["Source Name"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

#Standardize date
df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

#Standardize Reach
df["Reach"] = df["Reach"].fillna(0)

#Text preprocessing
def clean_text(text):
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["Cleaned_Content"] = df["Content"].apply(clean_text)

#Deduplication
before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["Cleaned_Content"]
)
after_duplicates = len(df)
print(f"Duplicate Records Removed: {before_duplicates - after_duplicates}")


#Validation checks
print("\nUnique Sentiment Values:")
print(df["Sentiment"].unique())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

#Output cleaned dataset

df.to_csv("data/processed_dataset.csv",index=False)

print("\nDATA CLEANING COMPLETED")
print(f"Final Records: {len(df)}")
print("Cleaned dataset saved as:")
print("data/processed_dataset.csv")