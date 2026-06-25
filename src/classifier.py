import pandas as pd
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv("data/processed_dataset.csv")

def classify_driver_rule_based(text):
    """
    Rule-based classification as fallback when ML model is not available.
    """
    text = str(text).lower()
    
    if len(text.strip()) < 10:
        return "UNCLASSIFIED", 0.0, "Insufficient text"
    
    driver_keywords = {
        "Brand Perception": [
            ("thought leadership", 3), ("expert commentary", 3), 
            ("cio", 3), ("chief investment officer", 3), ("fund manager", 3),
            ("market outlook", 3), ("industry viewpoint", 3), 
            ("market view", 3), ("investment outlook", 3),
            ("interview", 2), ("op-ed", 3), ("analysis", 2), 
            ("insight", 2), ("perspective", 2), ("expert says", 2),
            ("rate cut", 2), ("market forecast", 2), ("economic outlook", 2),
            ("exclusive interview", 3), ("expert view", 2),
            ("market strategy", 2), ("investment strategy", 2),
            ("s naren", 3), ("sankaran naren", 3), ("naren", 2),
            ("new fund offer", 3), ("nfo", 3), ("product launch", 3),
            ("fund launch", 3), ("scheme launch", 3), ("sip", 2),
            ("systematic investment plan", 3), ("pricing", 2), 
            ("expense ratio", 2), ("asset allocation", 2),
            ("equity fund", 2), ("hybrid fund", 2), ("debt fund", 2),
            ("multi asset", 3), ("aggressive hybrid", 3),
            ("small cap", 2), ("large cap", 2), ("mid cap", 2),
            ("retirement fund", 3), ("children fund", 3), 
            ("business cycle", 3), ("value fund", 2),
            ("festive offer", 3), ("positioning", 2),
            ("index fund", 2), ("etf", 2), ("exchange traded fund", 2),
            ("conglomerate fund", 3), ("momentum fund", 3),
            ("nifty private bank", 3), ("nifty next 50", 3),
            ("isif", 3), ("long short fund", 3),
            ("campaign", 3), ("sponsorship", 3), ("cricket sponsorship", 3),
            ("brand ambassador", 3), ("ad campaign", 3), 
            ("awareness initiative", 3), ("investor awareness", 3),
            ("marketing", 1), ("event", 2), ("promotion", 2),
            ("financial literacy drive", 3), ("outreach", 2)
        ],
        "User Experience": [
            ("fund returns", 3), ("scheme performance", 3),
            ("benchmark performance", 3), ("reliability", 2),
            ("outperform", 3), ("underperform", 3),
            ("cagr", 2), ("return generator", 2),
            ("top performer", 3), ("best performing", 2),
            ("quality", 1), ("feature", 1),
            ("down less than", 3), ("market downturn", 2),
            ("decline", 2), ("drop", 2), ("fall", 2), ("rise", 2),
            ("performance", 2), ("returns", 2),
            ("complaint", 3), ("customer service", 3),
            ("support", 2), ("helpline", 2),
            ("redemption", 2), ("kyc process", 3),
            ("complaint resolution", 3), ("issue resolved", 3),
            ("quick resolution", 3), ("delayed redemption", 3),
            ("slow kyc", 3), ("unresponsive helpline", 3),
            ("customer support", 3), ("issue", 2), ("problem", 2),
            ("app", 2), ("mobile app", 3), ("website", 2),
            ("digital experience", 3), ("onboarding", 3),
            ("transaction", 2), ("app crash", 3), 
            ("website downtime", 3), ("confusing screen", 3),
            ("smooth onboarding", 3), ("digital service", 3),
            ("online platform", 2), ("user interface", 2),
            ("login", 2), ("registration", 2), ("navigation", 2),
            ("app experience", 3), ("digital", 1)
        ],
        "Responsible Business Practices": [
            ("sebi", 3), ("regulatory", 3), ("regulatory action", 3),
            ("compliance", 2), ("governance", 2), 
            ("ethical governance", 3), ("ethics", 2),
            ("penalty", 3), ("disclosure lapse", 3),
            ("mis-selling", 3), ("mis selling", 3), 
            ("transparent governance", 3), ("regulator", 2),
            ("sebi gives green light", 3), ("ipo", 2),
            ("csr", 3), ("corporate social responsibility", 3),
            ("community engagement", 3), ("community", 2),
            ("financial literacy", 3), ("financial literacy drive", 3),
            ("donation", 3), ("relief activity", 3),
            ("women investor", 3), ("women-investor programme", 3),
            ("rural outreach", 3), ("social impact", 2),
            ("outreach initiative", 2), ("csr initiative", 3)
        ]
    }
    
    scores = {}
    match_details = {}
    
    for driver, keywords in driver_keywords.items():
        score = 0
        matched_keywords = []
        for keyword, weight in keywords:
            if keyword in text:
                score += weight
                matched_keywords.append((keyword, weight))
        scores[driver] = score
        match_details[driver] = matched_keywords
    
    total_score = sum(scores.values())
    if total_score == 0:
        if any(word in text for word in ["fund", "scheme", "investment", "market"]):
            return "Brand Perception", 0.2, "Context-based classification"
        return "UNCLASSIFIED", 0.0, "No keywords matched"
    
    best_driver = max(scores, key=scores.get)
    best_score = scores[best_driver]
    
    ratio_confidence = best_score / total_score if total_score > 0 else 0
    unique_matches = len(set([kw for kw, _ in match_details[best_driver]]))
    match_confidence = min(unique_matches / 2, 1.0)
    text_length = len(text.split())
    coverage = min(best_score / (text_length * 0.15), 1.0)
    
    confidence = (ratio_confidence * 0.5 + match_confidence * 0.3 + coverage * 0.2)
    confidence = min(confidence, 1.0)
    
    if confidence < 0.2:
        return "UNCLASSIFIED", confidence, "Low confidence"
    
    return best_driver, confidence, match_details[best_driver]

class MLClassifier:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.vectorizer = None
        self.model = None
        self.is_trained = False
        self.label_encoder = None
        
    def train(self, texts, labels):
        """Train ML model on labeled data"""
        print("\n" + "="*60)
        print("TRAINING ML CLASSIFIER")
        print("="*60)
        
        # Convert labels to numeric
        unique_labels = sorted(set(labels))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}
        y_numeric = [self.label_encoder[label] for label in labels]
        
        # Vectorize
        print("Vectorizing text...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        X = self.vectorizer.fit_transform(texts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
        )
        
        # Train model
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n Model Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        self.save_model()
        return accuracy
    
    def predict(self, texts):
        """Predict driver for new texts"""
        if not self.is_trained:
            self.load_model()
        
        X = self.vectorizer.transform(texts)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        confidences = [max(prob) for prob in probabilities]
        labels = [self.label_decoder[pred] for pred in predictions]
        return labels, confidences
    
    def save_model(self):
        """Save model and vectorizer"""
        joblib.dump(self.model, f"{self.model_dir}classifier.pkl")
        joblib.dump(self.vectorizer, f"{self.model_dir}vectorizer.pkl")
        joblib.dump(self.label_encoder, f"{self.model_dir}label_encoder.pkl")
        joblib.dump(self.label_decoder, f"{self.model_dir}label_decoder.pkl")
        print(f"\n Model saved to {self.model_dir}")
    
    def load_model(self):
        """Load saved model"""
        try:
            self.model = joblib.load(f"{self.model_dir}classifier.pkl")
            self.vectorizer = joblib.load(f"{self.model_dir}vectorizer.pkl")
            self.label_encoder = joblib.load(f"{self.model_dir}label_encoder.pkl")
            self.label_decoder = joblib.load(f"{self.model_dir}label_decoder.pkl")
            self.is_trained = True
            print(f" ML model loaded from {self.model_dir}")
            return True
        except FileNotFoundError:
            print(" ML model not found. Use rule-based classification.")
            return False

class HybridClassifier:
    def __init__(self):
        self.ml_classifier = MLClassifier()
        self.ml_available = self.ml_classifier.load_model()
        self.rule_based_fallback = True
        
        if self.ml_available:
            print(" Hybrid Classifier: ML mode active")
        else:
            print(" Hybrid Classifier: Rule-based mode active")
    
    def classify(self, text):
        """Classify using ML if available, fallback to rules"""
        text = str(text).strip()
        
        if len(text) < 10:
            return 'UNCLASSIFIED', 0.0, 'Text too short'
        
        # Try ML first
        if self.ml_available:
            try:
                pred, conf = self.ml_classifier.predict([text])
                pred = pred[0]
                conf = conf[0]
                
                if conf > 0.7:
                    return pred, conf, 'ML'
                elif conf > 0.5:
                    return pred, conf, 'ML (medium)'
            except Exception as e:
                print(f"ML error: {e}")
        
        # Fallback to rule-based
        if self.rule_based_fallback:
            pred, conf, details = classify_driver_rule_based(text)
            if pred != 'UNCLASSIFIED':
                return pred, conf, 'Rule-based'
        
        return 'UNCLASSIFIED', 0.0, 'No classification'

def classify_subdriver(text):
    """Classifies text into sub-drivers using rule-based approach."""
    text = str(text).lower()
    
    if len(text.strip()) < 10:
        return "UNCLASSIFIED", 0.0, "Insufficient text"
    
    subdriver_keywords = {
        "Thought Leadership": [
            ("cxo", 3), ("executive commentary", 3), ("fund manager", 3),
            ("cio", 3), ("chief investment officer", 3),
            ("market outlook", 3), ("industry viewpoint", 3),
            ("investment outlook", 3), ("market view", 3),
            ("expert commentary", 3), ("op-ed", 3),
            ("interview", 2), ("rate cut", 2), ("market forecast", 2),
            ("economic outlook", 2), ("analysis", 2), ("insight", 2),
            ("perspective", 2), ("s naren", 3), ("sankaran naren", 3),
            ("naren", 2), ("exclusive", 2), ("expert", 2), ("view", 1),
            ("strategy", 2), ("outlook", 2)
        ],
        "Product Strategy": [
            ("product launch", 3), ("new fund offer", 3), ("nfo", 3),
            ("fund launch", 3), ("scheme launch", 3),
            ("positioning", 2), ("pricing", 2), ("expense ratio", 2),
            ("new sip plan", 3), ("festive offer", 3),
            ("equity fund", 2), ("debt fund", 2), ("hybrid fund", 2),
            ("small cap fund", 3), ("large cap fund", 3), ("mid cap fund", 3),
            ("multi asset", 3), ("aggressive hybrid", 3),
            ("retirement fund", 3), ("children fund", 3),
            ("business cycle fund", 3), ("value fund", 2),
            ("index fund", 2), ("etf", 2), ("exchange traded fund", 2),
            ("conglomerate fund", 3), ("momentum fund", 3),
            ("nifty private bank", 3), ("nifty next 50", 3),
            ("isif", 3), ("long short fund", 3),
            ("offering", 2), ("scheme", 2), ("sip", 2)
        ],
        "Brand Visibility & Marketing": [
            ("campaign", 3), ("ad campaign", 3), ("sponsorship", 3),
            ("cricket sponsorship", 3), ("brand ambassador", 3),
            ("ambassador announcement", 3), ("awareness initiative", 3),
            ("investor awareness", 3), ("investor awareness event", 3),
            ("marketing", 1), ("event", 2), ("promotion", 2),
            ("awareness", 2)
        ],
        "Product & Service Quality": [
            ("fund returns", 3), ("scheme performance", 3),
            ("benchmark performance", 3), ("vs benchmark", 3),
            ("reliability", 2), ("quality", 1),
            ("outperform", 3), ("underperform", 3),
            ("cagr", 2), ("top performer", 3),
            ("feature", 1), ("praised feature", 3), ("criticised feature", 3),
            ("down less than", 3), ("market downturn", 2),
            ("decline", 2), ("drop", 2), ("fall", 2), ("rise", 2),
            ("performance", 2), ("returns", 2), ("growth", 2)
        ],
        "Customer Support & Complaint Resolution": [
            ("delayed redemption", 3), ("slow kyc", 3),
            ("kyc process", 3), ("unresponsive helpline", 3),
            ("complaint", 3), ("customer service", 3),
            ("support", 2), ("helpline", 2), ("redemption", 2),
            ("quick resolution", 3), ("issue resolved", 3),
            ("complaint resolution", 3), ("issue", 2), ("problem", 2),
            ("delay", 2)
        ],
        "Digital & Omnichannel Experience": [
            ("app", 2), ("mobile app", 3), ("app crash", 3),
            ("website", 2), ("website downtime", 3),
            ("digital service", 3), ("digital experience", 3),
            ("smooth onboarding", 3), ("onboarding", 2),
            ("confusing screen", 3), ("transaction screen", 3),
            ("online platform", 2), ("user interface", 2),
            ("login", 2), ("registration", 2), ("navigation", 2),
            ("app experience", 3), ("digital", 1)
        ],
        "Regulatory Compliance & Ethical Governance": [
            ("sebi", 3), ("sebi penalty", 3), ("regulatory action", 3),
            ("regulatory", 2), ("compliance", 2), ("governance", 2),
            ("ethical governance", 3), ("transparent governance", 3),
            ("disclosure lapse", 3), ("mis-selling", 3),
            ("mis selling", 3), ("mis-selling allegation", 3),
            ("ethics", 2), ("penalty", 3), ("regulator", 2),
            ("sebi gives green light", 3), ("ipo", 2)
        ],
        "Social Impact & Community (CSR)": [
            ("csr", 3), ("corporate social responsibility", 3),
            ("financial literacy", 3), ("financial literacy drive", 3),
            ("donation", 3), ("relief activity", 3),
            ("women investor", 3), ("women-investor programme", 3),
            ("rural outreach", 3), ("outreach initiative", 3),
            ("community engagement", 3), ("community", 2),
            ("social impact", 2), ("csr initiative", 3)
        ]
    }
    
    scores = {}
    match_details = {}
    
    for subdriver, keywords in subdriver_keywords.items():
        score = 0
        matched_keywords = []
        for keyword, weight in keywords:
            if keyword in text:
                score += weight
                matched_keywords.append((keyword, weight))
        scores[subdriver] = score
        match_details[subdriver] = matched_keywords
    
    total_score = sum(scores.values())
    if total_score == 0:
        if any(word in text for word in ["fund", "scheme", "launch", "nfo"]):
            return "Product Strategy", 0.2, "Context-based classification"
        if any(word in text for word in ["market", "outlook", "cio", "manager"]):
            return "Thought Leadership", 0.2, "Context-based classification"
        return "UNCLASSIFIED", 0.0, "No keywords matched"
    
    best_subdriver = max(scores, key=scores.get)
    best_score = scores[best_subdriver]
    
    ratio_confidence = best_score / total_score if total_score > 0 else 0
    unique_matches = len(set([kw for kw, _ in match_details[best_subdriver]]))
    match_confidence = min(unique_matches / 2, 1.0)
    
    confidence = (ratio_confidence * 0.5 + match_confidence * 0.5)
    confidence = min(confidence, 1.0)
    
    if confidence < 0.2:
        return "UNCLASSIFIED", confidence, "Low confidence"
    
    return best_subdriver, confidence, match_details[best_subdriver]

def enhance_sentiment(text, original_sentiment):
    """Validates and enhances sentiment with improved detection."""
    text = str(text).lower()
    
    positive_indicators = [
        "best", "top", "excellent", "outstanding", "superior",
        "success", "successful", "growth", "profit", "gain",
        "launch", "innovative", "leading", "strong", "positive",
        "outperform", "rise", "up", "good", "great", "benefit",
        "reward", "wealth", "smart", "efficient", "smooth",
        "easy", "convenient", "reliable", "trusted", "consistent"
    ]
    
    negative_indicators = [
        "issue", "problem", "complaint", "delay", "fail",
        "penalty", "violation", "mis-selling", "scam",
        "crash", "downtime", "unresponsive", "slow",
        "loss", "decline", "drop", "fall", "down", "negative",
        "underperform", "poor", "bad", "warning", "worst",
        "hopeless", "frustrating", "unprofessional", "buggy",
        "annoying", "useless", "pathetic"
    ]
    
    positive_count = sum(1 for word in positive_indicators if word in text)
    negative_count = sum(1 for word in negative_indicators if word in text)
    
    if "launch" in text and "nfo" in text:
        positive_count += 2
    if "complaint" in text or "issue" in text:
        negative_count += 1
    
    if positive_count > negative_count + 2:
        return "Positive", 0.8
    elif negative_count > positive_count + 2:
        return "Negative", 0.8
    elif positive_count > negative_count:
        return "Positive", 0.6
    elif negative_count > positive_count:
        return "Negative", 0.6
    else:
        return original_sentiment, 0.4

def classify_record(text):
    """Complete classification pipeline using Hybrid approach."""
    text = str(text).strip()
    
    if len(text) < 10:
        return {
            'driver': 'UNCLASSIFIED',
            'driver_confidence': 0.0,
            'driver_match_details': 'Text too short',
            'subdriver': 'UNCLASSIFIED',
            'subdriver_confidence': 0.0,
            'subdriver_match_details': 'Text too short',
            'validation_status': 'INVALID',
            'validation_messages': ['Text too short'],
            'sentiment_confidence': 0.0,
            'classification_method': 'N/A'
        }
    
    hybrid = HybridClassifier()
    driver, driver_conf, method = hybrid.classify(text)
    subdriver, subdriver_conf, subdriver_details = classify_subdriver(text)
    
    # If driver is UNCLASSIFIED but subdriver has matches, infer driver
    if driver == "UNCLASSIFIED" and subdriver != "UNCLASSIFIED":
        subdriver_to_driver = {
            "Thought Leadership": "Brand Perception",
            "Product Strategy": "Brand Perception",
            "Brand Visibility & Marketing": "Brand Perception",
            "Product & Service Quality": "User Experience",
            "Customer Support & Complaint Resolution": "User Experience",
            "Digital & Omnichannel Experience": "User Experience",
            "Regulatory Compliance & Ethical Governance": "Responsible Business Practices",
            "Social Impact & Community (CSR)": "Responsible Business Practices"
        }
        if subdriver in subdriver_to_driver:
            driver = subdriver_to_driver[subdriver]
            driver_conf = 0.3
            method = "Inferred from subdriver"
    
    # If driver has matches but subdriver is UNCLASSIFIED, infer subdriver
    if driver != "UNCLASSIFIED" and subdriver == "UNCLASSIFIED":
        driver_to_subdriver = {
            "Brand Perception": ["Thought Leadership", "Product Strategy", "Brand Visibility & Marketing"],
            "User Experience": ["Product & Service Quality", "Customer Support & Complaint Resolution", "Digital & Omnichannel Experience"],
            "Responsible Business Practices": ["Regulatory Compliance & Ethical Governance", "Social Impact & Community (CSR)"]
        }
        if driver in driver_to_subdriver:
            for possible in driver_to_subdriver[driver]:
                temp_sub, temp_conf, temp_details = classify_subdriver(text)
                if temp_sub != "UNCLASSIFIED":
                    subdriver = temp_sub
                    subdriver_conf = temp_conf * 0.7
                    subdriver_details = temp_details
                    break
    
    validation_messages = []
    if driver != "UNCLASSIFIED" and subdriver != "UNCLASSIFIED":
        validation_messages.append("Classification completed")
        validation_status = "VALID"
    else:
        validation_messages.append("Partial classification")
        validation_status = "REVIEW_NEEDED"
    
    return {
        'driver': driver,
        'driver_confidence': round(driver_conf, 2),
        'driver_match_details': f"Method: {method}",
        'subdriver': subdriver,
        'subdriver_confidence': round(subdriver_conf, 2),
        'subdriver_match_details': subdriver_details,
        'validation_status': validation_status,
        'validation_messages': ' | '.join(validation_messages),
        'sentiment_confidence': 0.0,
        'classification_method': method
    }

def train_ml_model():
    """Train the ML model using existing classified data."""
    print("\n" + "="*60)
    print("ML MODEL TRAINING SETUP")
    print("="*60)
    
    # Load data - use final_classified_dataset.csv for training
    df = pd.read_csv("data/final_classified_dataset.csv")
    
    # Use Content column directly (already cleaned and combined)
    # Filter out UNCLASSIFIED
    df = df[df["Driver"] != "UNCLASSIFIED"]
    
    if len(df) < 10:
        print(" Not enough classified data for ML training. Need at least 10 records.")
        return False
    
    # Get texts from Content column and labels
    texts = df["Content"].fillna("").tolist()
    labels = df["Driver"].tolist()
    
    print(f"Training data: {len(texts)} records")
    print(f"Unique drivers: {set(labels)}")
    
    ml_classifier = MLClassifier()
    accuracy = ml_classifier.train(texts, labels)
    
    print(f"\n ML Model trained with {accuracy:.2%} accuracy!")
    print("Model saved for future use.")
    return True

print("\n" + "="*60)
print("STARTING CLASSIFICATION")
print("="*60)

# Try to load ML model
hybrid = HybridClassifier()
ml_available = hybrid.ml_available

if not ml_available:
    print("\n ML model not found. Training new model...")
    train_ml_model()
    hybrid = HybridClassifier()
    ml_available = hybrid.ml_available

print(f"\nML Model Status: {' Available' if ml_available else '⚠️ Not Available'}")
print("Using Hybrid approach: ML + Rule-based fallback")

# Apply classification using Content column directly
classification_results = df["Content"].apply(classify_record)

# Extract results
df["Driver"] = classification_results.apply(lambda x: x['driver'])
df["Driver_Confidence"] = classification_results.apply(lambda x: x['driver_confidence'])
df["Sub driver"] = classification_results.apply(lambda x: x['subdriver'])
df["Sub_Driver_Confidence"] = classification_results.apply(lambda x: x['subdriver_confidence'])
df["Validation_Status"] = classification_results.apply(lambda x: x['validation_status'])
df["Validation_Messages"] = classification_results.apply(lambda x: x['validation_messages'])
df["Classification_Method"] = classification_results.apply(lambda x: x.get('classification_method', 'N/A'))

# Enhance sentiment
sentiment_results = df.apply(
    lambda row: enhance_sentiment(row["Content"], row["Sentiment"]),
    axis=1
)
df["Sentiment"] = sentiment_results.apply(lambda x: x[0])
df["Sentiment_Confidence"] = sentiment_results.apply(lambda x: x[1])

print("\n" + "="*60)
print("CLASSIFICATION STATISTICS")
print("="*60)

print("\nDriver Distribution:")
print(df["Driver"].value_counts())
print(f"\nClassification Rate: {len(df[df['Driver'] != 'UNCLASSIFIED'])}/{len(df)} = {len(df[df['Driver'] != 'UNCLASSIFIED'])/len(df)*100:.1f}%")

print("\nClassification Method Distribution:")
print(df["Classification_Method"].value_counts())

print("\nSub-Driver Distribution:")
print(df["Sub driver"].value_counts())

print("\nValidation Status:")
print(df["Validation_Status"].value_counts())

print("\nSentiment Distribution:")
print(df["Sentiment"].value_counts())

print("\nConfidence Statistics:")
print(f"Average Driver Confidence: {df['Driver_Confidence'].mean():.2f}")
print(f"Average Sub-Driver Confidence: {df['Sub_Driver_Confidence'].mean():.2f}")

unclassified = df[df["Driver"] == "UNCLASSIFIED"]
print(f"\nUnclassified Records: {len(unclassified)}")

print("\n" + "="*60)
print("SAMPLE CLASSIFICATIONS")
print("="*60)

sample_cols = ["Title", "Driver", "Driver_Confidence", "Classification_Method", "Sub driver", "Sentiment"]

high_conf = df[df["Driver_Confidence"] > 0.7]
if len(high_conf) > 0:
    print("\nHigh Confidence Samples (Confidence > 0.7):")
    print(high_conf[sample_cols].head(5))

medium_conf = df[(df["Driver_Confidence"] > 0.4) & (df["Driver_Confidence"] <= 0.7)]
if len(medium_conf) > 0:
    print("\nMedium Confidence Samples (0.4 < Confidence <= 0.7):")
    print(medium_conf[sample_cols].head(3))

column_order = [
    'Date', 'URL', 'Source Name', 'Title', 'Opening Text', 
    'Hit Sentence', 'Content',
    'Driver', 'Sub driver', 'Sentiment', 'Reach',
    'Driver_Confidence', 'Sub_Driver_Confidence', 'Sentiment_Confidence',
]

final_columns = [col for col in column_order if col in df.columns]
df_final = df[final_columns]

# Save
df_final.to_csv("data/final_classified_dataset.csv", index=False)

print("\n" + "="*60)
print("CLASSIFICATION COMPLETED")
print("="*60)
print(f"Total Records Processed: {len(df_final)}")
print(f"Classified Records: {len(df_final[df_final['Driver'] != 'UNCLASSIFIED'])}")
print(f"Unclassified Records: {len(df_final[df_final['Driver'] == 'UNCLASSIFIED'])}")
print(f"Classification Rate: {len(df_final[df_final['Driver'] != 'UNCLASSIFIED'])/len(df_final)*100:.1f}%")
print("\nFinal dataset saved as: data/final_classified_dataset.csv")