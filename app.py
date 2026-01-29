import streamlit as st
import nltk
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.tokenize import (
    word_tokenize,
    sent_tokenize,
    blankline_tokenize,
    WhitespaceTokenizer,
    WordPunctTokenizer
)
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, FreqDist, ngrams
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# ------------------ DOWNLOAD NLTK DATA ------------------
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ------------------ LOAD SPACY ------------------
nlp = spacy.load("en_core_web_sm")

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="NLP Text Analyzer", layout="wide")
st.title("🧠 NLP Text Analyzer using Streamlit")
st.write("Paste a paragraph below to explore **tokenization, NLP features, and embeddings**")

text = st.text_area("✍️ Enter your paragraph here", height=200)

if text:
    # ================= TOKENIZATION =================
    st.header("🔹 Tokenization")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Word Tokenization")
        st.write(word_tokenize(text))

        st.subheader("Sentence Tokenization")
        st.write(sent_tokenize(text))

        st.subheader("Blank Line Tokenization")
        st.write(blankline_tokenize(text))

    with col2:
        st.subheader("Whitespace Tokenization")
        st.write(WhitespaceTokenizer().tokenize(text))

        st.subheader("WordPunct Tokenization")
        st.write(WordPunctTokenizer().tokenize(text))

    # ================= N-GRAMS =================
    st.header("🔹 N-Grams")

    tokens = word_tokenize(text.lower())

    st.subheader("Bigrams")
    st.write(list(ngrams(tokens, 2)))

    st.subheader("Trigrams")
    st.write(list(ngrams(tokens, 3)))

    st.subheader("N-Grams (Custom)")
    n = st.slider("Select N value", 1, 5, 2)
    st.write(list(ngrams(tokens, n)))

    # ================= STEMMING & LEMMATIZATION =================
    st.header("🔹 Stemming & Lemmatization")

    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    stemmed = [ps.stem(word) for word in tokens]
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stemming")
        st.write(stemmed)

    with col2:
        st.subheader("Lemmatization")
        st.write(lemmatized)

    # ================= TOP WORDS =================
    st.header("🔹 Top Frequent Words")

    freq_dist = FreqDist(tokens)
    top_words = freq_dist.most_common(10)
    st.table(pd.DataFrame(top_words, columns=["Word", "Frequency"]))

    # ================= POS TAGGING =================
    st.header("🔹 Part-of-Speech (POS) Tagging")
    pos_tags = pos_tag(tokens)
    st.write(pos_tags)

    # ================= NER =================
    st.header("🔹 Named Entity Recognition (NER)")

    doc = nlp(text)
    ner_data = [(ent.text, ent.label_) for ent in doc.ents]
    if ner_data:
        st.table(pd.DataFrame(ner_data, columns=["Entity", "Label"]))
    else:
        st.info("No named entities found.")

    # ================= WORD CLOUD =================
    st.header("🔹 Word Cloud")

    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

    # ================= EMBEDDINGS =================
    st.header("🔹 Text Embeddings")

    # ---- Bag of Words ----
    st.subheader("Bag of Words (BoW)")
    bow = CountVectorizer()
    bow_matrix = bow.fit_transform([text])
    st.dataframe(pd.DataFrame(
        bow_matrix.toarray(),
        columns=bow.get_feature_names_out()
    ))

    # ---- TF-IDF ----
    st.subheader("TF-IDF")
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([text])
    st.dataframe(pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf.get_feature_names_out()
    ))

    # ---- Word2Vec ----
    st.subheader("Word2Vec Embeddings")
    sentences = [tokens]
    w2v = Word2Vec(sentences, vector_size=50, window=5, min_count=1)

    word = st.selectbox("Select a word", list(w2v.wv.index_to_key))
    st.write(f"Vector for '{word}':")
    st.write(w2v.wv[word])
