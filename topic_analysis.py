from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
import pandas as pd
import re


df = pd.read_csv('dati/dataseset_italia_con_sentiment.csv', header=None, skiprows=1)

# Esempio di dati (sostituisci con il tuo dataset)
documents = df[0].astype(str)

# Tokenizzazione e Preprocessamento
def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Rimuovi caratteri non alfabetici
    text = text.lower()  # Converti il testo in minuscolo
    if len(text) > 512:
        text = text[:512]
    return text

documents = [preprocess(doc) for doc in documents]

# Analisi del Sentiment
sentiment_analyzer = pipeline("sentiment-analysis")
sentiments = [sentiment_analyzer(doc)[0]['label'].upper() for doc in documents]

# Preparazione dei dati per LDA
vectorizer = CountVectorizer(stop_words='english')  # Rimuovi le stop words
X = vectorizer.fit_transform(documents)

# Modello LDA
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X)

# Estrazione dei topic principali
feature_names = vectorizer.get_feature_names_out()
topics = []
for topic_idx, topic in enumerate(lda.components_):
    top_keywords_idx = topic.argsort()[:-5-1:-1]
    top_keywords = [feature_names[i] for i in top_keywords_idx]
    topics.append(top_keywords)

# Stampa dei risultati
for i, (doc, sentiment, topic) in enumerate(zip(documents, sentiments, topics), 1):
    print(f"Documento {i}:")
    print(f"Sentiment: {sentiment}")
    print(f"Topic: {topic}")
    print(f"Contenuto: {doc}")
    print("="*50)