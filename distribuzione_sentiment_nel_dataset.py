import pandas as pd
import matplotlib.pyplot as plt

# Caricamento del dataset
file_path = 'dati/dataseset_italia_con_sentiment.csv'
df = pd.read_csv(file_path, header=None, skiprows=1)

# Assumi che la colonna 4 sia il sentiment
sentiments = df[4]

# Conteggio della frequenza di ciascun valore di sentiment
sentiment_counts = sentiments.value_counts()
print(sentiment_counts)

# Creazione del grafico a barre
plt.figure(figsize=(10, 6))
sentiment_counts.plot(kind='bar', color='skyblue')
plt.title('Distribuzione dei Valori di Sentiment nel Dataset')
plt.xlabel('Sentiment')
plt.ylabel('Conteggio dei commenti')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()
