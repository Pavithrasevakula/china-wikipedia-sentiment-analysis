# China Wikipedia Sentiment Analysis

This project analyzes the sentiment of content from the Wikipedia page about China, using natural language processing and machine learning techniques.

## Project Overview

The project scrapes the Wikipedia page about China, processes the text, and performs sentiment analysis on the sentences. It then builds machine learning models to predict sentiment and provides a Streamlit web application for users to analyze new sentences.

## Project Structure

```
china-wikipedia-sentiment-analysis
│
├── sentiment_china_project_code.ipynb   # Jupyter notebook with the full analysis pipeline
├── app.py                               # Streamlit web application
├── best_model.pkl                       # Saved machine learning model 
├── tfidf_vectorizer.pkl                 # Saved TF-IDF vectorizer 
└── china_wordcloud.png                  # Word cloud visualization 
```

## Features

- Web scraping Wikipedia content using BeautifulSoup
- Text preprocessing and cleaning
- Sentiment analysis using TextBlob
- Word cloud generation for visualization
- TF-IDF feature extraction for text data
- Multiple machine learning models for sentiment classification
- SMOTE technique for handling class imbalance
- Streamlit web application for interactive sentiment analysis

## Installation

1. Clone this repository:
```
git clone https://github.com/Pavithrasevakula/china-wikipedia-sentiment-analysis.git
cd china-wikipedia-sentiment-analysis
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Run the Jupyter notebook to generate the model files:
```
jupyter notebook sentiment_china_project_code.ipynb
```

4. Launch the Streamlit app:
```
streamlit run app.py
```

## Usage

1. Run the Jupyter notebook to understand the data processing and model training pipeline.
2. Use the Streamlit app to analyze the sentiment of sentences about China:
   - Enter a sentence in the text area
   - Click on "Analyze Sentiment"
   - View the sentiment prediction and probability

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk
- textblob
- beautifulsoup4
- requests
- wordcloud
- imbalanced-learn
- streamlit

## Author

- Pavithra Sevakula