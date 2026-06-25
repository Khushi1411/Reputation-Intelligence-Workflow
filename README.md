# Reputation Intelligence Workflow

A comprehensive reputation intelligence workflow for analyzing digital mentions of ICICI Prudential Asset Management Company. This solution demonstrates a complete workflow from data processing to intelligent classification and interactive visualization.

##  Overview

This project implements an end-to-end reputation intelligence workflow that processes, analyzes, classifies, and visualizes data of digital mentions related to a BFSI
brand from various sources including:
- News websites (The Economic Times, CNBC-TV18, Business Today)
- Social media (LinkedIn, Reddit)
- App reviews
- Online forums 

The workflow is designed to be **scalable** and **automated**, with no manual classification required.

### Key Features

-  Overview Analytics: Total mentions, sentiment distribution, driver and sub-driver analysis
-  Content Explorer: Search and filter by driver, sub-driver, and sentiment
-  Insights: Positive and negative reputation drivers with percentage breakdowns
-  ML-Powered Classification: Hybrid approach using Machine Learning + Rule-based fallback
-  Interactive Visualizations: Pie charts, bar charts, and keyword analysis
-  Scalable Workflow: Can process unlimited records with ML model

### Classification Framework

| Reputation Driver | Sub-Parameters |
|-------------------|----------------|
| **Brand Perception** | Thought Leadership, Product Strategy, Brand Visibility & Marketing |
| **User Experience** | Product & Service Quality, Customer Support & Complaint Resolution, Digital & Omnichannel Experience |
| **Responsible Business Practices** | Regulatory Compliance & Ethical Governance, Social Impact & Community (CSR) |

---
### Live Demo

The dashboard is deployed and accessible at:
**https://reputation-intelligence-workflow-abxvolv8ya3xha7jjxmchh.streamlit.app/**

---

##  Project Structure

```
reputation-intelligence-workflow/
│
├── app.py                          # Main Streamlit dashboard
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── data/
│   ├── Dataset.xlsx                # Original dataset
│   ├── processed_dataset.csv       # Cleaned dataset
│   └── final_classified_dataset.csv # Classified dataset
│
├── src/
│   ├── dataset_processing.py       # Data cleaning pipeline
│   └── classifier.py               # ML + Rule-based classification engine
│
├── models/
    ├── classifier.pkl              # Trained ML model
    ├── vectorizer.pkl              # TF-IDF vectorizer
    ├── label_encoder.pkl           # Label encoder
    └── label_decoder.pkl           # Label decoder
```

##  Setup Instructions

###  Prerequisites

* Python 3.10 or higher
* pip (Python package manager)

---

##  Local Installation

### 1️ Clone the Repository

```bash
git clone https://github.com/Khushi1411/Reputation-Intelligence-Workflow.git
cd reputation-intelligence-workflow
```

---

### 2️ Create Virtual Environment (Recommended)

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️ Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Run the Workflow

The workflow consists of two main steps:

---

###  Step 1: Data Cleaning & Processing

```bash
python src/dataset_processing.py
```

####  This step performs:

* Creates `content` column from Title, Opening Text, and Hit Sentence
* Removes irrelevant (non-BFSI) records
* Standardizes:

  * Sentiment
  * Source name
  * Date
  * Reach
* Text preprocessing:

  * Lowercasing
  * URL removal
  * Special character removal
* Removes duplicate records

####  Output:

```
data/processed_dataset.csv
```

---

###  Step 2: ML-Powered Classification

```bash
python src/classifier.py
```

####  This step performs:

* First run: Trains ML model (Random Forest)
* Subsequent runs: Uses trained model
* Hybrid approach:

  * ML model (primary)
  * Rule-based fallback
* Classifies:

  * Reputation Drivers
  * Sub-Drivers
* Enhances sentiment analysis

####  Output:

```
data/final_classified_dataset.csv
```
---

###  Step 3: Launch dashboard

```bash
streamlit run app.py
```

---
