import streamlit as st
import nltk
import spacy
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import (
    word_tokenize,
    sent_tokenize,
    blankline_tokenize,
    WhitespaceTokenizer,
    WordPunctTokenizer
)
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag, FreqDist, ngrams
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(page_title="NLP Text Analyzer", layout="wide")
st.title("🧠 NLP Text Analyzer using Streamlit")

# --------------------------------------------------
# SAFE NLTK DOWNLOAD (NO CACHING)
# --------------------------------------------------
def download_nltk():
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download("stopwords")

download_nltk()

# --------------------------------------------------
# CACHE SPACY MODEL
# --------------------------------------------------
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
text = st.text_area("✍️ Enter your paragraph", height=200)

if st.button("🔍 Run NLP Analysis") and text:
    st.session_state.analyzed = True
    st.session_state.text = text

    tokens = word_tokenize(text.lower())
    st.session_state.tokens = tokens

    # Tokenization
    st.session_state.word_tokens = word_tokenize(text)
    st.session_state.sent_tokens = sent_tokenize(text)
    st.session_state.blank_tokens = blankline_tokenize(text)
    st.session_state.white_tokens = WhitespaceTokenizer().tokenize(text)
    st.session_state.wordpunct_tokens = WordPunctTokenizer().tokenize(text)

    # Stopwords
    stop_words = set(stopwords.words("english"))
    st.session_state.stopwords_removed = [
        w for w in tokens if w.isalpha() and w not in stop_words
    ]

    # NLP
    st.session_state.freq = FreqDist(tokens)
    st.session_state.pos = pos_tag(tokens)
    st.session_state.doc = nlp(text)

    # N-grams
    st.session_state.bigrams = list(ngrams(tokens, 2))
    st.session_state.trigrams = list(ngrams(tokens, 3))

    # Embeddings
    st.session_state.w2v = Word2Vec(
        [tokens],
        vector_size=50,
        window=5,
        min_count=1
    )

    # BoW & TF-IDF
    bow = CountVectorizer()
    tfidf = TfidfVectorizer()

    st.session_state.bow = bow.fit_transform([text])
    st.session_state.bow_vocab = bow.get_feature_names_out()

    st.session_state.tfidf = tfidf.fit_transform([text])
    st.session_state.tfidf_vocab = tfidf.get_feature_names_out()

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------
if st.session_state.analyzed:

    tokens = st.session_state.tokens

    # ================= TOKENIZATION =================
    st.header("🔹 Tokenization ")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Word Tokenization**")
        st.write(st.session_state.word_tokens)
        st.success(f"Total words: {len(st.session_state.word_tokens)}")

        st.write("**Sentence Tokenization**")
        st.write(st.session_state.sent_tokens)
        st.success(f"Total sentences: {len(st.session_state.sent_tokens)}")

        st.write("**Blank Line Tokenization**")
        st.write(st.session_state.blank_tokens)
        st.success(f"Total blank-line tokens: {len(st.session_state.blank_tokens)}")

    with col2:
        st.write("**Whitespace Tokenization**")
        st.write(st.session_state.white_tokens)
        st.success(f"Total whitespace tokens: {len(st.session_state.white_tokens)}")

        st.write("**WordPunct Tokenization**")
        st.write(st.session_state.wordpunct_tokens)
        st.success(f"Total wordpunct tokens: {len(st.session_state.wordpunct_tokens)}")

    # ================= STOPWORDS =================
    st.header("🔹 Stopwords Removal ")

    st.write("**Tokens after stopwords removal**")
    st.write(st.session_state.stopwords_removed)

    st.info(f"Total tokens before stopwords: {len(tokens)}")
    st.success(f"Total tokens after stopwords: {len(st.session_state.stopwords_removed)}")

    # ================= N-GRAMS =================
    st.header("🔹 Types Of Tokens ")

    st.write("**Bigrams**")
    st.write(st.session_state.bigrams)
    st.success(f"Total bigrams: {len(st.session_state.bigrams)}")

    st.write("**Trigrams**")
    st.write(st.session_state.trigrams)
    st.success(f"Total trigrams: {len(st.session_state.trigrams)}")

    n = st.slider("Select custom N value", 1, 5, 2)
    custom_ngrams = list(ngrams(tokens, n))
    st.write(f"**{n}-Grams**")
    st.write(custom_ngrams)
    st.success(f"Total {n}-grams: {len(custom_ngrams)}")

    # ================= STEM & LEMMA =================
    st.header("🔹 Stemming & Lemmatization")

    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Stemming**")
        st.write([ps.stem(w) for w in tokens])

    with col2:
        st.write("**Lemmatization**")
        st.write([lemmatizer.lemmatize(w) for w in tokens])

    # ================= TOP WORDS =================
    st.header("🔹 Top Frequent Words")
    st.table(
        pd.DataFrame(
            st.session_state.freq.most_common(10),
            columns=["Word", "Frequency"]
        )
    )

    # ================= POS =================
    st.header("🔹 POS Tagging")
    st.write(st.session_state.pos)

    # ================= NER =================
    st.header("🔹 Named Entity Recognition")
    ner = [(ent.text, ent.label_) for ent in st.session_state.doc.ents]
    if ner:
        st.table(pd.DataFrame(ner, columns=["Entity", "Label"]))
    else:
        st.info("No named entities found")

    # ================= WORD CLOUD =================
    with st.expander("🔹 Word Cloud"):
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(st.session_state.text)

        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

    # ================= BoW =================
    st.header("🔹 Bag of Words (BoW)")
    st.dataframe(
        pd.DataFrame(
            st.session_state.bow.toarray(),
            columns=st.session_state.bow_vocab
        )
    )
    st.success(f"Total BoW features: {len(st.session_state.bow_vocab)}")

    # ================= TF-IDF =================
    st.header("🔹 TF-IDF")
    st.dataframe(
        pd.DataFrame(
            st.session_state.tfidf.toarray(),
            columns=st.session_state.tfidf_vocab
        )
    )
    st.success(f"Total TF-IDF features: {len(st.session_state.tfidf_vocab)}")

    # ================= WORD2VEC =================
    st.header("🔹 Word2Vec Embeddings")
    w2v = st.session_state.w2v
    word = st.selectbox("Select a word", list(w2v.wv.index_to_key))
    st.write(w2v.wv[word])
