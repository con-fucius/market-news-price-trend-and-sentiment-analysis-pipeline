# Install required packages
!pip install openai langchain requests pandas nltk matplotlib pyLDAvis gensim

import openai
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
import nltk

# Initialize OpenAI API key
openai.api_key = 'your-openai-api-key'

# NewsAPI endpoint and API key (for additional news source)
NEWS_API_KEY = 'your-news-api-key'
NEWS_API_URL = "https://newsapi.org/v2/everything"

# CurrentsAPI (alternative source)
CURRENTS_API_KEY = 'your-currents-api-key'
CURRENTS_API_URL = "https://api.currentsapi.services/v1/search"

# Fetch Blockchain News from NewsAPI
def fetch_blockchain_news():
    params = {
        'q': 'blockchain',
        'language': 'en',
        'sortBy': 'publishedAt',
        'apiKey': NEWS_API_KEY
    }
    response = requests.get(NEWS_API_URL, params=params)
    data = response.json()
    if data.get('status') == 'ok':
        return data['articles']
    else:
        return []

# Fetch Blockchain News from Currents API (Alternative source)
def fetch_blockchain_news_currents():
    params = {
        'keywords': 'blockchain',
        'language': 'en',
        'apiKey': CURRENTS_API_KEY
    }
    response = requests.get(CURRENTS_API_URL, params=params)
    data = response.json()
    if data.get('status') == 'ok':
        return data['news']
    else:
        return []

# Combine results from both APIs
def fetch_combined_news():
    news_from_newsapi = fetch_blockchain_news()
    news_from_currents = fetch_blockchain_news_currents()
    return news_from_newsapi + news_from_currents

# Summarize Articles Using OpenAI's LLM
llm = OpenAI(model="text-davinci-003", temperature=0.7)

def summarize_article(article):
    prompt = f"Summarize the following article:\n{article['title']}\n{article['description']}\n{article['content']}"
    summary = llm(prompt)
    return summary

# Sentiment Analysis Using GPT-3.5-turbo for Sentiment
sentiment_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def analyze_sentiment(text):
    prompt = f"Determine the sentiment of the following text. The sentiment can be either Positive, Negative, or Neutral:\n{text}"
    response = sentiment_llm([HumanMessage(content=prompt)])
    sentiment = response.content.strip().lower()
    return sentiment

# Process News Articles to Summarize and Analyze Sentiment
def process_articles(articles):
    summarized_articles = [(article['title'], summarize_article(article)) for article in articles]
    sentiments = [(title, summarize, analyze_sentiment(summarize)) for title, summarize in summarized_articles]
    return summarized_articles, sentiments

# Topic Modeling (LDA)
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.isalpha() and word not in stop_words]

def perform_topic_modeling(texts, num_topics=3):
    processed_texts = [preprocess_text(text) for text in texts]
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topics = lda_model.print_topics(num_words=5)
    return topics

# Plotting Sentiment Distribution
def plot_sentiment_distribution(sentiments):
    sentiment_counts = pd.Series([s[2] for s in sentiments]).value_counts()
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
    plt.title("Sentiment Distribution of Blockchain News")
    plt.ylabel("Count")
    plt.xlabel("Sentiment")
    plt.show()

# Trend Analysis - Enhanced with Sentiment Weighting
def identify_trend(sentiment_counts):
    sentiment_weights = {'positive': 1, 'neutral': 0, 'negative': -1}
    weighted_sentiment = sum([sentiment_weights.get(sentiment, 0) for sentiment in sentiment_counts])
    
    if weighted_sentiment > 0:
        return "Bullish"
    elif weighted_sentiment < 0:
        return "Bearish"
    else:
        return "Neutral"

# Main Execution

# Fetch combined blockchain news
articles = fetch_combined_news()

# Summarize and analyze sentiment of articles
summarized_articles, sentiments = process_articles(articles)

# Extract text for topic modeling (use summaries for topic modeling)
texts_for_topic_modeling = [summarize for _, summarize in summarized_articles]
topics = perform_topic_modeling(texts_for_topic_modeling)

# Display identified topics
print("Identified Topics:")
for topic in topics:
    print(topic)

# Sentiment Distribution Visualization
plot_sentiment_distribution(sentiments)

# Trend Identification
sentiment_counts = pd.Series([s[2] for s in sentiments]).value_counts()
overall_trend = identify_trend(sentiment_counts)
print(f"Overall Market Sentiment: {overall_trend}")

