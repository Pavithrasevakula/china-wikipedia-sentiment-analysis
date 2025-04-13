import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import time
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="China Wikipedia Sentiment Analysis",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1; 
    }
    .info-text {
        font-size: 1rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0D47A1;
    }
</style>
""", unsafe_allow_html=True)

# Load the best model and TF-IDF vectorizer
@st.cache_resource
def load_models():
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer, True
    except FileNotFoundError:
        return None, None, False

# Function to get image download link
def get_image_download_link(fig, filename, text):
    buffered = BytesIO()
    fig.savefig(buffered, format="png", dpi=300, bbox_inches='tight')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Predict sentiment of a sentence
def predict_sentiment(model, vectorizer, sentence):
    # Transform the sentence using the loaded vectorizer
    sentence_vector = vectorizer.transform([sentence])
    
    # Get prediction probability
    prediction_proba = model.predict_proba(sentence_vector)[0]
    prediction = model.predict(sentence_vector)[0]
    
    # Get class labels
    class_labels = model.classes_
    
    # Create probability dictionary
    proba_dict = {label: prob for label, prob in zip(class_labels, prediction_proba)}
    
    # For comparison, get TextBlob sentiment
    textblob_analysis = TextBlob(sentence)
    textblob_polarity = textblob_analysis.sentiment.polarity
    textblob_subjectivity = textblob_analysis.sentiment.subjectivity
    
    if textblob_polarity > 0:
        textblob_sentiment = "positive"
    elif textblob_polarity < 0:
        textblob_sentiment = "negative"
    else:
        textblob_sentiment = "neutral"
    
    return prediction, proba_dict, textblob_polarity, textblob_subjectivity, textblob_sentiment

# Plot sentiment probability
def plot_sentiment_proba(proba_dict):
    # Create dataframe for visualization
    df = pd.DataFrame({
        'Sentiment': list(proba_dict.keys()),
        'Probability': list(proba_dict.values())
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Set colors
    colors = ['#d9534f', '#5cb85c'] if 'negative' in df['Sentiment'].iloc[0] else ['#5cb85c', '#d9534f']
    
    # Create bar plot
    sns.barplot(x='Sentiment', y='Probability', data=df, palette=colors, ax=ax)
    
    # Add value labels
    for i, v in enumerate(df['Probability']):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    # Set labels and title
    ax.set_title('Sentiment Prediction Probability', fontsize=15)
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1.1)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

# Main app
def main():
    st.markdown('<h1 class="main-header"> China Wikipedia Sentiment Analysis</h1>', unsafe_allow_html=True)
    
    # About section
    with st.expander("ℹ️ About this App"):
        st.markdown("""
        <div class="info-text">
        <p>This application analyzes the sentiment of sentences related to China using a machine learning model trained on Wikipedia data.</p>
        
        <p><b>Project Components:</b></p>
        <ol>
            <li>Web scraping the Wikipedia page of China</li>
            <li>Text preprocessing and cleaning</li>
            <li>Sentiment analysis using TextBlob</li>
            <li>Feature extraction using TF-IDF</li>
            <li>Model training with various classifiers</li>
            <li>Model evaluation and selection</li>
            <li>Streamlit application for user interaction</li>
        </ol>
        
        <p><b>Instructions:</b> Enter a sentence about China and click 'Analyze Sentiment' to see the prediction.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load models
    model, vectorizer, models_loaded = load_models()
    
    if not models_loaded:
        st.error("Model files not found. Please run the sentiment_analysis.py script first to generate the models.")
        st.stop()
    
    # Create columns for layout
    col1, col2 = st.columns([3, 2])
    
    # Input area
    with col1:
        st.markdown('<h2 class="sub-header">Enter Text for Analysis</h2>', unsafe_allow_html=True)
        sentence = st.text_area(
            "Type or paste a sentence about China:",
            "China has a rich cultural heritage that spans thousands of years.",
            height=150
        )
        
        analyze_button = st.button("🔍 Analyze Sentiment")
    
    # Results area
    with col2:
        st.markdown('<h2 class="sub-header">Results</h2>', unsafe_allow_html=True)
        
        if analyze_button and sentence:
            with st.spinner('Analyzing sentiment...'):
                # Add slight delay for better UX
                time.sleep(0.5)
                
                # Get predictions
                prediction, proba_dict, textblob_polarity, textblob_subjectivity, textblob_sentiment = predict_sentiment(model, vectorizer, sentence)
                
                # Display ML model results
                st.markdown("#### Machine Learning Model Prediction")
                
                # Display sentiment with appropriate styling
                if prediction == "positive":
                    st.success(f"Sentiment: POSITIVE")
                else:
                    st.error(f"Sentiment: NEGATIVE")
                
                # Plot and display probability
                fig = plot_sentiment_proba(proba_dict)
                st.pyplot(fig)
                
                # Add download link for the plot
                st.markdown(get_image_download_link(fig, "sentiment_prediction.png", "Download Plot"), unsafe_allow_html=True)
                
                # TextBlob results
                st.markdown("#### TextBlob Analysis")
                st.write(f"Polarity score: {textblob_polarity:.4f}")
                st.write(f"Subjectivity: {textblob_subjectivity:.4f}")
                
                if textblob_sentiment == "positive":
                    st.success(f"TextBlob sentiment: POSITIVE")
                elif textblob_sentiment == "negative":
                    st.error(f"TextBlob sentiment: NEGATIVE")
                else:
                    st.info(f"TextBlob sentiment: NEUTRAL")
    
    # Information about model
    st.markdown('<h2 class="sub-header">Model Information</h2>', unsafe_allow_html=True)
    
    model_col1, model_col2 = st.columns(2)
    
    with model_col1:
        st.markdown("""
        <div class="info-text">
        <b>Model Type:</b> The best model (highest F1 score) from the following:
        <ul>
            <li>Binary Logistic Regression</li>
            <li>Decision Tree</li>
            <li>Random Forest</li>
            <li>Gradient Boosting</li>
            <li>Naïve Bayes</li>
            <li>K Nearest Neighbors</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with model_col2:
        st.markdown("""
        <div class="info-text">
        <b>Data Source:</b> Wikipedia page on China<br>
        <b>Features:</b> TF-IDF vectors of sentences<br>
        <b>Target:</b> Binary sentiment (positive/negative)<br>
        <b>Balancing:</b> SMOTE technique to handle class imbalance<br>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <hr>
    <div style="text-align: center; color: #666;">
        <p>Created by Pavithra Sevakula</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()