from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import string
nltk.download('stopwords')
nltk.download('punkt')

# Funzione per rimuovere la punteggiatura
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    text_without_punctuation = text.translate(translator)
    return text_without_punctuation

# Funzione per rimuovere le stopwords
def remove_stopwords(text):
    stopwords = set(nltk.corpus.stopwords.words('english'))  # Cambia 'italian' in base alla lingua del tuo testo
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered_words)


# Caricamento del dataset
file_path = 'dati/dataseset_italia_con_sentiment.csv'
df = pd.read_csv(file_path, header=None, skiprows=1)

df[0] = df[0].astype(str)

df[0] = df[0].apply(remove_punctuation)
df[0] = df[0].apply(remove_stopwords)

# Separazione dei testi in base al sentiment (positivo e negativo)
positive_texts = df[df[4] == 'positive'][0].str.cat(sep=' ')
negative_texts = df[df[4] == 'negative'][0].str.cat(sep=' ')


nessun_sentiment = df[0].str.cat(sep=' ')

# Creazione di WordClouds per sentiment positivo e negativo
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_texts)
wordcloud_negative = WordCloud(width=800, height=400, background_color='black').generate(negative_texts)

wordcloud_nessuno = WordCloud(width=800, height=400, background_color='black').generate(nessun_sentiment)


# Visualizzazione delle WordClouds
plt.figure(figsize=(16, 8))


# WordCloud per sentiment positivo
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Parole Positive', fontsize=18)

# WordCloud per sentiment negativo
plt.subplot(1, 2, 2)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Parole Negative', fontsize=18)

'''
# WordCloud per sentiment positivo
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_nessuno, interpolation='bilinear')
plt.axis('off')
plt.title('Nessun setiment', fontsize=18)
'''

plt.show()
