# Module 1 – Data Pipeline

## Objective

This module builds a complete data engineering pipeline using the Books to Scrape website.

The pipeline performs:

- Web scraping
- Data cleaning
- Currency conversion
- SQLite database creation
- SQL analysis
- Pandas integration

---

## Data Source

https://books.toscrape.com

The website is a public website designed specifically for web scraping practice.

---

## Installation

Install the required Python packages:

```bash
pip install -r ../requirements.txt
```

---

## Running the Notebook

Open:

```
data_pipeline.ipynb
```

Run all cells from top to bottom.

The notebook performs:

1. Scrapes book data
2. Cleans the dataset
3. Converts prices to INR
4. Creates the SQLite database
5. Executes SQL queries
6. Demonstrates pandas SQL integration

---

## Dataset

The dataset contains books from three categories:

- Mystery
- Historical Fiction
- Thriller

Total books scraped:

**69**

---

## Data Cleaning Decisions

The following transformations were applied:

- Removed the £ symbol from prices.
- Converted prices to floating-point values.
- Converted star ratings (One–Five) into integer values (1–5).
- Converted stock availability into Boolean values.
- Numeric parsing failures were handled using median imputation to ensure the pipeline did not fail.

---

## Currency Conversion

A fixed project-defined conversion rate was used:

**1 GBP = 105.50 INR**

No external API or live exchange-rate service was used.

---

## Database Schema

### Categories

| Column | Type |
|--------|------|
| category_id | INTEGER PRIMARY KEY |
| category_name | TEXT UNIQUE |

### Books

| Column | Type |
|--------|------|
| book_id | INTEGER PRIMARY KEY |
| title | TEXT |
| price_gbp | REAL |
| price_inr | REAL |
| rating | INTEGER |
| in_stock | INTEGER |
| category_id | INTEGER (Foreign Key) |

---

## SQL Operations

The notebook demonstrates:

- SELECT
- WHERE
- ORDER BY
- LIMIT
- DISTINCT
- BETWEEN
- JOIN
- GROUP BY

---

## Pandas Operations

The notebook demonstrates:

- `pd.read_sql()`
- `pd.merge()`

The SQL JOIN result and the pandas merge result are equivalent.