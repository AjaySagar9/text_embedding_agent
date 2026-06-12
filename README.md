# 🤖 Text Embedding Agent using TF-IDF

A beginner-friendly NLP (Natural Language Processing) project that converts text into numerical vector representations (embeddings) using TF-IDF and performs semantic similarity search using Cosine Similarity.

This agent helps users find the most relevant document from a collection of documents based on their query.

---

# 📌 Project Overview

Computers cannot understand text directly.

To process text, it must first be converted into numerical vectors called **embeddings**.

This project demonstrates:

* Text Vectorization
* Text Embeddings
* Semantic Search
* Similarity Matching
* Information Retrieval

The agent converts documents and user queries into TF-IDF vectors and finds the most similar document.

---

# 🚀 Features

✅ Text Embedding Generation

✅ TF-IDF Vectorization

✅ Semantic Search

✅ Cosine Similarity Matching

✅ Real-Time Query Search

✅ Lightweight Implementation

✅ No API Key Required

✅ Google Colab Compatible

---

# 🛠️ Technologies Used

| Technology        | Purpose                   |
| ----------------- | ------------------------- |
| Python            | Programming Language      |
| Scikit-Learn      | Machine Learning Library  |
| TF-IDF Vectorizer | Text Embedding Generation |
| Cosine Similarity | Similarity Calculation    |
| Google Colab      | Development Environment   |

---

# 📂 Project Structure

```text
text-embedding-agent/
│
├── app.ipynb
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Install required library:

```bash
pip install scikit-learn
```

---

# 💻 Source Code

## Step 1: Create Document Dataset

```python
documents = [

    "Python is a programming language",

    "Machine Learning is a subset of Artificial Intelligence",

    "SQL is used to manage databases",

    "Deep Learning uses neural networks",

    "Data Science combines statistics and programming"

]
```

---

## Step 2: Generate Text Embeddings

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

document_embeddings = vectorizer.fit_transform(
    documents
)
```

---

## Step 3: Create Similarity Agent

```python
from sklearn.metrics.pairwise import cosine_similarity

def text_embedding_agent(query):

    query_embedding = vectorizer.transform(
        [query]
    )

    scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    best_index = scores.argmax()

    return (
        documents[best_index],
        scores[best_index]
    )
```

---

## Step 4: Run Agent

```python
while True:

    query = input(
        "Ask Something: "
    )

    if query.lower() == "exit":
        break

    result, score = text_embedding_agent(
        query
    )

    print("Best Match:", result)

    print(
        "Similarity Score:",
        round(score, 3)
    )
```

---

# 🔍 Complete Code Explanation

## What is TF-IDF?

TF-IDF stands for:

```text
TF = Term Frequency

IDF = Inverse Document Frequency
```

It measures the importance of a word within a collection of documents.

Words that appear frequently in one document but rarely in others receive higher importance.

---

## What is Text Embedding?

Text Embedding converts text into numerical vectors.

Example:

```text
Python Programming

↓

[0.21, 0.45, 0.78, ...]
```

Computers use these vectors instead of raw text.

---

## TfidfVectorizer

```python
vectorizer = TfidfVectorizer()
```

Purpose:

* Converts text into numerical vectors
* Creates document embeddings
* Prepares text for similarity calculations

---

## fit_transform()

```python
document_embeddings = vectorizer.fit_transform(
    documents
)
```

Purpose:

* Learns vocabulary
* Converts documents into embeddings

Output:

```text
Document 1 → Vector

Document 2 → Vector

Document 3 → Vector
```

---

## transform()

```python
query_embedding = vectorizer.transform(
    [query]
)
```

Purpose:

Converts user query into an embedding vector.

---

## Cosine Similarity

```python
cosine_similarity(
    query_embedding,
    document_embeddings
)
```

Purpose:

Measures similarity between vectors.

Range:

```text
0 → Completely Different

1 → Exactly Similar
```

---

# 🔄 Agent Workflow

```text
Documents
     │
     ▼
TF-IDF Vectorizer
     │
     ▼
Document Embeddings
     │
     ▼
User Query
     │
     ▼
Query Embedding
     │
     ▼
Cosine Similarity
     │
     ▼
Best Matching Document
     │
     ▼
Display Result
```

---

# 📊 Example

## Input

```text
Ask Something:
database language
```

---

## Output

```text
Best Match:

SQL is used to manage databases

Similarity Score:

0.63
```

---

## Another Example

### Input

```text
Ask Something:
artificial intelligence
```

### Output

```text
Best Match:

Machine Learning is a subset of Artificial Intelligence

Similarity Score:

0.81
```

---

# 🎯 Applications

* Search Engines
* Chatbots
* Question Answering Systems
* Semantic Search
* Information Retrieval
* Recommendation Systems
* Document Search
* RAG Foundations

---

# Advantages

* Fast Processing
* Lightweight
* Easy to Understand
* No External APIs
* No Heavy Models
* Beginner Friendly

---

# Limitations

* Keyword-Based Matching
* Limited Semantic Understanding
* Less Powerful than Transformer Models
* Small Dataset Dependency

---

# Future Improvements

* Sentence Transformers
* Hugging Face Embeddings
* Vector Databases
* FAISS Integration
* RAG Systems
* Streamlit UI
* PDF Knowledge Base
* Resume Screening Integration

---

# Learning Outcomes

After completing this project, you will understand:

* Text Embeddings
* TF-IDF
* Cosine Similarity
* Semantic Search
* NLP Fundamentals
* Information Retrieval
* AI Agent Development

---

# Resume Description

Developed a Text Embedding Agent using Python and TF-IDF Vectorization. The system converts documents and user queries into numerical embeddings, performs semantic similarity search using cosine similarity, and retrieves the most relevant document based on user input.

---

# 👨‍💻 Author

## Ajay Sagar

* Python Developer
* AI & Machine Learning Enthusiast
* B.Tech CSE Student

### Connect With Me

🔗 LinkedIn: https://www.linkedin.com/in/engineerajay

🚀 Building AI Projects and Learning New Technologies Every Day.

**Code the Future with Us..!**
