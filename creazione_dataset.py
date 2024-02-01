from nltk.sentiment import SentimentIntensityAnalyzer
import json
import pandas as pd
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
def semantic_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores

def sentiment(compound):
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'
def remove_punctuation(text):
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"  # Lista dei caratteri di punteggiatura da rimuovere
    text_without_punctuation = "".join(i for i in text if i not in punctuation)
    return text_without_punctuation

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
    keywords = word_freq.most_common(5)

    return [keyword[0] for keyword in keywords]





# Carica i dati dal file JSON
with open('dati/dati_sub_reddit_unitedkingdom.json', 'r', encoding='utf-8') as file: data = json.load(file)

#keywords extraction
data_to_append = []

for post in data:
    # Estrai le informazioni dal post
    title = str(post['Titolo'])
    post_text = str(post['Testo del post'])
    date = post['Data']

    # Rimuovi le stopwords dal titolo e estrai le parole chiave
    no_stopwords_title = remove_stopwords(title)
    extraction_key = keywords_extraction(no_stopwords_title)

    # Analizza il titolo e il testo del post
    title_scores = semantic_analysis(title)
    post_scores = semantic_analysis(post_text)

    # Aggiungi i dati alla lista
    data_to_append.append({'Frase': title, 'Parole chiave': extraction_key, 'Data': date, 'Sentiment': title_scores, 'Sentiment risultato': sentiment(title_scores['compound'])})
    data_to_append.append({'Frase': post_text, 'Parole chiave': extraction_key, 'Data': date, 'Sentiment': post_scores, 'Sentiment risultato': sentiment(post_scores['compound'])})

    # Itera sui commenti
    for comment in post['Commenti']:
        commento = str(comment['Testo del commento'])
        data_commento = comment['Data del commento']

        if commento is not None and commento != "[deleted]":
            no_stopwords_comment = remove_stopwords(commento)
            extraction_key_comment = keywords_extraction(no_stopwords_comment)
        # Anlisi semantica del commento
            comment_scores = semantic_analysis(commento)
            data_to_append.append({'Frase': commento, 'Parole chiave': extraction_key_comment, 'Data': data_commento, 'Sentiment': comment_scores, 'Sentiment risultato': sentiment(comment_scores['compound'])})
        else:
            continue
        # Itera sulle risposte
        for response in comment['Risposte']:
            risposta = str(response['Testo della risposta'])
            data_risposta = response['Data della risposta']

            if risposta is not None and risposta != "[deleted]" or risposta != "[removed]":
                no_stopwords_response = remove_stopwords(risposta)
                extraction_key_response = keywords_extraction(no_stopwords_response)
                response_scores = semantic_analysis(risposta)
                data_to_append.append({'Frase': risposta, 'Parole chiave': extraction_key_response, 'Data': data_risposta, 'Sentiment': response_scores, 'Sentiment risultato': sentiment(response_scores['compound'])})
            else:
                continue

# Crea un nuovo DataFrame utilizzando i dati nella lista
df = pd.DataFrame(data_to_append)

# Rimuovi i record che contengono "[deleted]" nella colonna 0
df = df[df.iloc[:, 0] != ('')]
df = df[df.iloc[:, 0] != '[deleted]']
df = df[df.iloc[:, 0] != '[removed]']


# Salva il DataFrame in formato CSV
df.to_csv('dati/dataseset_unitedkingdom_con_sentiment.csv', index=False)