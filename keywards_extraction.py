import json
import pandas as pd
import nltk
nltk.download('vader_lexicon')

def remove_punctuation(text):
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"  # Lista dei caratteri di punteggiatura da rimuovere
    text_without_punctuation = "".join(i for i in text if i not in punctuation)
    return text_without_punctuation

#Rimozione delle stopwords
def remove_stopwords(text):
    if not isinstance(text, str):
        raise ValueError("Il testo deve essere una stringa")
    stopwords = set(nltk.corpus.stopwords.words('english'))
    rm_punctuation  = remove_punctuation(text)
    words = nltk.word_tokenize(rm_punctuation)
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return filtered_words

def keywords_extraction(words):
    word_freq = nltk.FreqDist(words)

    # Estrazione delle parole chiave più comuni
    keywords = word_freq.most_common(10)

    return [keyword[0] for keyword in keywords]


# Carica i dati dal file CSV
df = pd.read_csv('dati/dataseset_italia_con_sentiment.csv', header=None)

#keywords extraction


all_keywords = []


for sentence in df[0]:  # Assumendo che le frasi siano nella prima colonna
    if not isinstance(sentence, str):
        continue
    cleaned_text = remove_stopwords(sentence)
    keywords = keywords_extraction(cleaned_text)
    all_keywords.extend(keywords)

# Se necessario, ottenere parole chiave uniche e più frequenti
unique_keywords = keywords_extraction(all_keywords)

print(unique_keywords)