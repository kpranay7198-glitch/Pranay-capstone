# Capstone Project

## Overview

This repository contains three integrated modules demonstrating a complete data and machine-learning workflow:

1. **Module 1 — Data Pipeline**
2. **Module 2 — Analytics Pipeline**
3. **Module 3 — Zepto Support Assistant**

The project progresses from data acquisition and database engineering to exploratory analysis and predictive modeling, and finally to a production-style RAG-based support application.

---

# Project Architecture

```text
                    CAPSTONE PROJECT
                          |
        +-----------------+-----------------+
        |                 |                 |
        v                 v                 v
   MODULE 1           MODULE 2           MODULE 3
 Data Pipeline     Analytics Pipeline   Support Assistant
        |                 |                 |
        v                 v                 v
 Web Scraping       Titanic Dataset     Zepto Policies
        |                 |                 |
        v                 v                 v
 Data Cleaning       EDA + Modeling     Embeddings
        |                 |                 |
        v                 v                 v
 SQLite Database     ML Pipeline        ChromaDB
        |                 |                 |
        v                 v                 v
 SQL + Pandas        Evaluation         LangGraph
                                           |
                                           v
                                         FastAPI
                                           |
                                           v
                                         Docker
```

---

# Module 1 — Data Pipeline

## Objective

Module 1 builds a complete data engineering pipeline using the **Books to Scrape** website.

The pipeline performs:

* Web scraping
* Data cleaning
* Currency conversion
* SQLite database creation
* SQL analysis
* Pandas integration

---

## Data Source

**Books to Scrape**

```text
https://books.toscrape.com
```

The website is a public website specifically designed for web-scraping practice.

The scraping pipeline uses Python web-scraping tools to collect structured book information.

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Notebook

Open:

```text
data_pipeline.ipynb
```

Run all cells from top to bottom.

The notebook performs:

1. Scrapes book data.
2. Cleans the dataset.
3. Converts GBP prices to INR.
4. Creates the SQLite database.
5. Executes SQL queries.
6. Demonstrates Pandas SQL integration.

---

## Dataset

The dataset contains books from three categories:

* Mystery
* Historical Fiction
* Thriller

### Total Books

**69 books**

---

## Data Cleaning

The following transformations were applied:

* Removed the `£` symbol from prices.
* Converted prices to floating-point values.
* Converted star ratings from words (`One`–`Five`) into integers (`1`–`5`).
* Converted stock availability into Boolean values.
* Numeric parsing failures were handled using median imputation to ensure pipeline robustness.

---

## Currency Conversion

A fixed project-defined exchange rate was used:

```text
1 GBP = 105.50 INR
```

No external API or live exchange-rate service was used.

The INR price is calculated from the cleaned GBP price using this fixed rate.

---

## Database

SQLite was used as the relational database.

### Categories Table

| Column        | Type                |
| ------------- | ------------------- |
| category_id   | INTEGER PRIMARY KEY |
| category_name | TEXT UNIQUE         |

### Books Table

| Column      | Type                |
| ----------- | ------------------- |
| book_id     | INTEGER PRIMARY KEY |
| title       | TEXT                |
| price_gbp   | REAL                |
| price_inr   | REAL                |
| rating      | INTEGER             |
| in_stock    | INTEGER             |
| category_id | INTEGER FOREIGN KEY |

The schema separates categories from books and connects them through `category_id`.

---

## SQL Operations

The notebook demonstrates:

* `SELECT`
* `WHERE`
* `ORDER BY`
* `LIMIT`
* `DISTINCT`
* `BETWEEN`
* `JOIN`
* `GROUP BY`

These operations demonstrate filtering, sorting, aggregation, uniqueness checks, range queries, and relational joins.

---

## Pandas Integration

The notebook demonstrates:

```python
pd.read_sql()
```

for loading SQL query results into Pandas.

It also demonstrates:

```python
pd.merge()
```

for performing the equivalent relational merge in Pandas.

The SQL `JOIN` result and Pandas merge result are equivalent.

---

# Module 2 — Analytics Pipeline

## Objective

Module 2 demonstrates a complete analytics and machine-learning workflow using the **Titanic dataset**.

The workflow includes:

* Dataset profiling
* Data cleaning
* Exploratory Data Analysis
* Feature preprocessing
* Predictive modeling
* Model evaluation
* Class imbalance analysis
* Hyperparameter tuning
* Regression analysis
* Residual analysis
* Final pipeline deployment artifact

---

## Dataset

**Source:** Seaborn Titanic Dataset

| Property |      Value |
| -------- | ---------: |
| Records  |        891 |
| Features |         15 |
| Target   | `survived` |

The Titanic dataset was loaded once using Seaborn's built-in dataset loader and saved as:

```text
titanic.csv
```

All subsequent analysis was performed using the saved dataset.

---

# Part A — Exploratory Data Analysis

## Dataset Profiling

The dataset was profiled using:

```python
df.info()
df.describe()
df.shape
```

These methods were used to inspect:

* Dataset dimensions
* Data types
* Missing values
* Statistical properties

---

## Missing Value Handling

The following threshold-based strategy was applied:

| Missing Percentage | Strategy              |
| ------------------ | --------------------- |
| < 5%               | Drop affected rows    |
| 5%–30%             | Impute missing values |
| > 30%              | Remove column         |

### Applied Strategies

| Column   | Missing % | Strategy          |
| -------- | --------: | ----------------- |
| Age      |    19.87% | Median Imputation |
| Embarked |     0.22% | Drop Missing Rows |
| Deck     |    77.22% | Column Removed    |

### Rationale

**Age:** Missingness falls between 5% and 30%, so median imputation preserves observations while reducing sensitivity to outliers.

**Embarked:** Missingness is below 5%, so affected rows were removed.

**Deck:** Missingness exceeds 30%, making reliable imputation impractical.

---

# Univariate Analysis

## Age

A histogram and box plot were created.

Outliers were identified using the IQR method.

**Total Age Outliers: 65**

---

## Fare

A histogram and box plot were created.

Outliers were identified using the IQR method.

**Total Fare Outliers: 114**

---

## Fare Distribution

Mean, median, and mode were calculated.

The relationship:

```text
Mean > Median > Mode
```

indicates that Fare is **positively/right skewed**.

This indicates that a small number of passengers paid substantially higher fares than most passengers.

---

# Bivariate Analysis

## Survival by Gender

Female passengers had a significantly higher survival rate than male passengers, indicating that gender was an important factor in survival.

---

## Survival by Passenger Class

First-class passengers had the highest survival rate, while third-class passengers had the lowest survival rate.

This demonstrates the importance of passenger class.

---

## Survival by Gender and Passenger Class

Female passengers travelling in first class had the highest survival probability.

Male passengers travelling in third class had the lowest survival probability.

This demonstrates the combined influence of gender and socioeconomic status.

---

# Correlation Analysis

The correlation matrix used:

* `survived`
* `pclass`
* `age`
* `sibsp`
* `parch`
* `fare`

The derived boolean variables:

* `adult_male`
* `alone`

were excluded because they are redundant features derived from other variables.

### Important Relationships

#### Fare ↔ Passenger Class

A strong negative correlation exists because the numerical encoding of passenger class increases toward third class, while higher fares are generally associated with first class.

#### SibSp ↔ Parch

A positive correlation exists because passengers travelling with siblings often also travelled with parents or children.

---

# Multivariate Analysis

## Age vs Survival

Younger passengers generally showed slightly higher survival rates, although age alone was not a decisive predictor.

## Fare vs Survival

Passengers paying higher fares had a considerably greater likelihood of survival.

This reinforces the relationship between fare, passenger class, and survival.

## Embarked vs Survival

Passengers embarking from Cherbourg showed comparatively higher survival rates.

## Pair Plot

The pair plot demonstrated relationships among numerical variables.

Fare and passenger class showed clearer separation between survivors and non-survivors than age.

---

# Standardization

`Age` and `Fare` were standardized using:

```python
StandardScaler
```

The transformed features achieved approximately:

```text
Mean ≈ 0
Standard Deviation ≈ 1
```

confirming successful standardization.

---

# Part B — Predictive Modeling

## Train-Test Split

A **stratified train-test split** was performed before preprocessing.

Stratification preserves the target class distribution across training and testing datasets.

This produces a more reliable evaluation.

---

# Preprocessing Pipeline

Preprocessing was fitted **only on the training data** to prevent data leakage.

The pipeline includes:

* Median Imputation
* Most Frequent Imputation
* One-Hot Encoding
* StandardScaler
* ColumnTransformer
* Scikit-learn Pipeline

---

# Classification Models

Three classifiers were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

## Classification Results

| Model               |   Accuracy |  Precision |     Recall |         F1 |        AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression | **80.90%** | **0.7833** | **0.6912** | **0.7344** | **0.8610** |
| Decision Tree       | **76.97%** | **0.6901** | **0.7206** | **0.7050** | **0.7541** |
| Random Forest       | **82.02%** | **0.7813** | **0.7353** | **0.7576** | **0.8179** |

---

# Class Imbalance Analysis

Three approaches were compared:

* Baseline Random Forest
* `class_weight='balanced'`
* SMOTE

| Method       |  Precision |     Recall |         F1 |
| ------------ | ---------: | ---------: | ---------: |
| Baseline     | **0.7813** | **0.7353** | **0.7576** |
| Class Weight | **0.7391** | **0.7500** | **0.7445** |
| SMOTE        | **0.7460** | **0.6912** | **0.7176** |

The baseline Random Forest produced the highest F1 score.

Class Weight slightly increased recall but reduced precision and overall F1.

SMOTE produced the lowest F1 score.

Therefore, the **baseline Random Forest** was selected as the preferred model.

---

# Hyperparameter Tuning

`GridSearchCV` was used to optimize the Random Forest classifier.

### Best Parameters

The exact values should be filled from the final notebook output:

```text
n_estimators = <your value>
max_depth = <your value>
max_features = <your value>
```

### OOB Score

```text
<your OOB score>
```

These values should be replaced with the actual results before final submission.

---

# Regression Analysis

A multivariate Linear Regression model was trained to predict passenger Fare.

| Model             |          MAE |          RMSE |          R² |          Adjusted R² |
| ----------------- | -----------: | ------------: | ----------: | -------------------: |
| Linear Regression | `<your MAE>` | `<your RMSE>` | `<your R²>` | `<your Adjusted R²>` |

The exact regression values should be taken directly from the final notebook.

---

# Residual Analysis

The residual plot showed an increasing spread of residuals as predicted Fare increased.

This indicates **heteroscedasticity**, meaning that the variance of prediction errors is not constant across the prediction range.

---

# Final Classification Recommendation

The **Random Forest classifier** achieved the strongest overall classification performance:

* Accuracy: **82.02%**
* Recall: **73.53%**
* F1 Score: **75.76%**

Logistic Regression achieved the highest AUC:

**0.8610**

However, Random Forest provided the best overall balance between precision and recall and therefore was selected as the preferred deployment model.

---

# Module 2 Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Joblib

---

# Module 3 — Zepto Support Assistant

## Objective

Module 3 implements an offline customer-support assistant using a **Retrieval-Augmented Generation (RAG)** architecture.

The system uses:

* 8 Zepto policy documents
* `all-MiniLM-L6-v2`
* ChromaDB
* LangGraph
* Pydantic
* FastAPI
* Uvicorn
* Docker
* `MOCK_LLM=1`

The system distinguishes between policy-related questions and general questions.

---

# Module 3 Architecture

```text
                         User Query
                             |
                             v
                    +------------------+
                    | classify_intent  |
                    +------------------+
                       /            \
                      /              \
                     v                v
          policy_question       general_question
                  |                    |
                  v                    v
       +---------------------+   +---------------+
       | retrieve_and_answer |   | direct_answer |
       +---------------------+   +---------------+
                  |
                  v
          Query Embedding
                  |
                  v
             ChromaDB
                  |
                  v
             Top-3 Sources
                  |
                  v
          Structured Response
                  |
                  v
              FastAPI
                  |
                  v
               Docker
```

---

# RAG Pipeline

```text
8 TXT Documents
       |
       v
Document Ingestion
       |
       v
all-MiniLM-L6-v2
       |
       v
Embeddings
       |
       v
ChromaDB
       |
       v
User Query
       |
       v
Query Embedding
       |
       v
Top-3 Retrieval
       |
       v
Answer Generation
       |
       v
FastAPI /ask
```

---

# Module 3 Corpus

| Document     | Policy                   |
| ------------ | ------------------------ |
| `doc_01.txt` | Delivery Policy          |
| `doc_02.txt` | Returns & Refunds        |
| `doc_03.txt` | Membership Tiers         |
| `doc_04.txt` | Order Tracking           |
| `doc_05.txt` | Order Cancellation       |
| `doc_06.txt` | Damaged or Missing Items |
| `doc_07.txt` | Gift Cards               |
| `doc_08.txt` | Customer Support Hours   |

---

# Embedding Model

The project uses:

```text
all-MiniLM-L6-v2
```

The embedding model runs locally.

No external embedding API is required.

---

# ChromaDB

ChromaDB is used as the vector database.

The ingestion process:

1. Loads all 8 documents.
2. Generates embeddings.
3. Creates the `zepto_policies` collection.
4. Stores document embeddings and metadata.

Successful ingestion:

```text
Loaded 8 documents.

Ingestion complete!
Collection: zepto_policies
Documents stored: 8
```

---

# Retrieval Verification

A test query was:

```text
How much does Zepto charge for delivery?
```

The top retrieved document was:

```text
doc_01
```

This is the correct source because `doc_01` contains the Delivery Policy.

---

# Structured Prompt

The structured prompt is implemented in:

```text
support_assistant/prompt.py
```

It contains:

* Role
* Context
* Task
* Format
* Length
* Negative constraint
* Few-shot example

The response schema is:

```json
{
  "answer": "string",
  "sources": ["string"],
  "confidence": 0.0
}
```

---

# LangGraph

The LangGraph implementation is contained in:

```text
support_assistant/main.py
```

It contains three required nodes:

### `classify_intent`

Classifies the question as:

```text
policy_question
```

or:

```text
general_question
```

### `retrieve_and_answer`

Performs:

* Query embedding
* ChromaDB retrieval
* Top-3 document retrieval
* Context construction
* Answer generation

### `direct_answer`

Handles general questions that do not concern Zepto policies.

---

# Mock LLM Mode

The graded baseline uses:

```text
MOCK_LLM=1
```

This does not require an external LLM API or API key.

For a policy question, the system returns an answer beginning with:

```text
Based on the retrieved context:
```

For a general question:

```text
I can only answer questions about Zepto policies right now.
```

---

# Pydantic Validation

The response is validated using:

```python
class SupportResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
```

The confidence score is constrained between:

```text
0.0 and 1.0
```

The implementation also contains corrective retry logic for the optional real-LLM path.

The validation flow supports up to two corrective retries before returning a clear validation error.

---

# FastAPI

The API exposes:

```text
POST /ask
```

Example request:

```json
{
  "query": "How much does delivery cost?"
}
```

---

# Local API Test

Start the server:

```powershell
python -m uvicorn support_assistant.main:app --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

### Policy Query

```powershell
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method Post -ContentType "application/json" -Body '{"query":"How much does delivery cost?"}'
$response | ConvertTo-Json
```

Successful response:

```json
{
    "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
    "sources": [
        "doc_01",
        "doc_05",
        "doc_02"
    ],
    "confidence": 1.0
}
```

### General Query

```powershell
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method Post -ContentType "application/json" -Body '{"query":"What is the capital of India?"}'
$response | ConvertTo-Json
```

Successful response:

```json
{
    "answer": "I can only answer questions about Zepto policies right now.",
    "sources": [],
    "confidence": 1.0
}
```

---

# Docker

The project includes a Dockerfile for containerized deployment.

The container:

1. Installs Python dependencies.
2. Copies the application.
3. Runs document ingestion.
4. Creates the ChromaDB database.
5. Starts Uvicorn.

### Build

From the project root:

```powershell
docker build -t zepto-support .
```

### Run

```powershell
docker run --rm -p 7860:7860 zepto-support
```

The container exposes:

```text
http://127.0.0.1:7860
```

The Docker container was successfully tested with:

```text
Loaded 8 documents.
Documents stored: 8
Application startup complete.
Uvicorn running on http://0.0.0.0:7860
```

---

# Docker API Verification

### Policy Query

```powershell
$response = Invoke-RestMethod -Uri "http://127.0.0.1:7860/ask" -Method Post -ContentType "application/json" -Body '{"query":"How much does delivery cost?"}'
$response | ConvertTo-Json
```

Response:

```json
{
    "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
    "sources": [
        "doc_01",
        "doc_05",
        "doc_02"
    ],
    "confidence": 1.0
}
```

### General Query

```powershell
$response = Invoke-RestMethod -Uri "http://127.0.0.1:7860/ask" -Method Post -ContentType "application/json" -Body '{"query":"What is the capital of India?"}'
$response | ConvertTo-Json
```

Response:

```json
{
    "answer": "I can only answer questions about Zepto policies right now.",
    "sources": [],
    "confidence": 1.0
}
```

---

# Module 3 Technologies

* Python 3.11
* Sentence Transformers
* all-MiniLM-L6-v2
* ChromaDB
* LangGraph
* Pydantic
* FastAPI
* Uvicorn
* Docker

---

# Complete Project Structure

```text
Pranay-capstone/
│
├── README.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
│
├── Module 1/
│   └── data_pipeline.ipynb
│
├── Module 2/
│   └── analytics/
│       ├── 01_eda.ipynb
│       ├── 02_modeling.ipynb
│       ├── titanic.csv
│       ├── best_random_forest_pipeline.joblib
│       └── figures/
│
└── Module 3/
    └── support_assistant/
        ├── __init__.py
        ├── main.py
        ├── ingest.py
        ├── retrieve.py
        ├── prompt.py
        ├── requirements.txt
        │
        ├── docs/
        │   ├── doc_01.txt
        │   ├── doc_02.txt
        │   ├── doc_03.txt
        │   ├── doc_04.txt
        │   ├── doc_05.txt
        │   ├── doc_06.txt
        │   ├── doc_07.txt
        │   └── doc_08.txt
        │
        └── data/
            └── chroma/
```

> Adjust the folder names in this tree to match your actual repository structure before submission.

---

# Overall Technologies

| Area                | Technologies            |
| ------------------- | ----------------------- |
| Programming         | Python                  |
| Web Scraping        | Requests, BeautifulSoup |
| Data Processing     | Pandas, NumPy           |
| Database            | SQLite                  |
| Visualization       | Matplotlib, Seaborn     |
| Machine Learning    | Scikit-learn            |
| Imbalanced Learning | Imbalanced-learn        |
| Model Persistence   | Joblib                  |
| Embeddings          | Sentence Transformers   |
| Vector Database     | ChromaDB                |
| Workflow            | LangGraph               |
| Validation          | Pydantic                |
| API                 | FastAPI, Uvicorn        |
| Containerization    | Docker                  |
| Notebooks           | Jupyter                 |

---

# Final Summary

The capstone demonstrates an end-to-end progression across three major data and AI engineering stages.

### Module 1

A complete **data engineering pipeline** was developed using web scraping, data cleaning, currency conversion, SQLite normalization, SQL analysis, and Pandas integration.

### Module 2

A complete **analytics and machine-learning pipeline** was developed using the Titanic dataset, including EDA, preprocessing, classification, class-imbalance comparison, Random Forest tuning, regression, residual analysis, and model persistence.

### Module 3

A complete **offline RAG customer-support system** was developed using local embeddings, ChromaDB, LangGraph, Pydantic, FastAPI, and Docker.

The final project therefore demonstrates:

```text
Data Acquisition
       ↓
Data Engineering
       ↓
Database Design
       ↓
Data Analysis
       ↓
Machine Learning
       ↓
Model Evaluation
       ↓
RAG / AI Application
       ↓
API
       ↓
Dockerized Deployment
```

This provides a complete progression from raw data acquisition to an operational machine-learning/AI application.
