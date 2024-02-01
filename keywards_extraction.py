import json
import pandas as pd
import nltk
nltk.download('vader_lexicon')



# Carica i dati dal file CSV
df = pd.read_csv('dati/dataseset_italia_con_sentiment.csv', header=None)

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


#Estrazione delle parole chiave
def keywords_extraction(words):
    word_freq = nltk.FreqDist(words)
    # Estrazione delle parole chiave più comuni
    keywords = word_freq.most_common(400)
    return [keyword[0] for keyword in keywords]


all_keywords = []

#Estrazione delle parole chiave da ogni frase
for sentence in df[0].astype(str):
    cleaned_text = remove_stopwords(sentence)
    keywords = keywords_extraction(cleaned_text)
    all_keywords.extend(keywords)

#Dopo aver estratto tutte le parole chiave dalle singole frasi, rimuove quelle duplicate
unique_keywords = keywords_extraction(all_keywords)

print(unique_keywords)