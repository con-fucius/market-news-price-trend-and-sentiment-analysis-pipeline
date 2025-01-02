# Blockchain News Analysis with AI

This repository provides a Python script for analyzing blockchain-related news using APIs, AI-based text summarization, sentiment analysis, and topic modeling. It combines tools like OpenAI, LangChain, and Gensim to extract insights and visualize trends in the blockchain space.

---

## Features

- **Fetch Blockchain News**:
  - Retrieves articles from **NewsAPI** and **CurrentsAPI**.
  - Combines articles from both sources for broader coverage.

- **Summarization**:
  - Summarizes articles using OpenAI's GPT-based language models.

- **Sentiment Analysis**:
  - Analyzes the sentiment (Positive, Negative, Neutral) of the summarized articles.

- **Topic Modeling**:
  - Uses Latent Dirichlet Allocation (LDA) to extract key topics from news articles.

- **Visualization**:
  - Visualizes sentiment distribution with Matplotlib.
  - Identifies overall market trends based on sentiment analysis.

---

## Installation

### Prerequisites

Ensure you have Python installed and install the required packages:

```bash
pip install openai langchain requests pandas nltk matplotlib pyLDAvis gensim
```

### Additional Setup

- **API Keys**:
  - Add your `OpenAI`, `NewsAPI`, and `CurrentsAPI` keys to the script

- **Download NLTK Data**:
  The script uses NLTK for tokenization and stopword removal. Ensure the required datasets are downloaded:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  ```

---

## Usage

Run the script to perform the following:

1. Fetch and combine blockchain-related news articles from NewsAPI and CurrentsAPI.
2. Summarize articles using OpenAI's language model.
3. Analyze sentiment for each article summary.
4. Perform topic modeling to identify key themes in the news.
5. Visualize sentiment distribution.
6. Identify the overall market trend (Bullish, Bearish, Neutral) based on sentiment.

---


## License

This project is open-source and licensed under the MIT License. Feel free to use, modify, and share!

---

## Acknowledgments

- [OpenAI](https://openai.com/)
- [NewsAPI](https://newsapi.org/)
- [CurrentsAPI](https://currentsapi.services/)
- [NLTK](https://www.nltk.org/)
- [Gensim](https://radimrehurek.com/gensim/)

--- 
