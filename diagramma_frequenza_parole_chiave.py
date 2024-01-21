import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

dataset = pd.read_csv('dati/dataseset_italia_con_sentiment.csv', header=None, skiprows=1)

# Parole chiave da cercare
keywords = ['right', 'life', 'law', 'suicide', 'allow', 'favor', 'suffering', 'autonomy', 'decision', 'free', 'euthanasia']

# Estrazione delle frasi dalla colonna "Frase"
frasi = dataset[0].dropna().tolist()

# Conteggio delle frequenze delle parole chiave
frequenze = Counter(word for frase in frasi for word in frase.split() if word in keywords)

# Ordina le parole chiave in base alla frequenza (dalla più alta alla più bassa)
frequenze = dict(sorted(frequenze.items(), key=lambda item: item[1], reverse=True))

# Creazione del diagramma a barre per visualizzare le frequenze
plt.figure(figsize=(10, 6))
plt.bar(frequenze.keys(), frequenze.values(), color='skyblue')
plt.xlabel('Parole Chiave')
plt.ylabel('Frequenza')
plt.title('Frequenza delle Parole Chiave nel Dataset (Ordinate)')
plt.xticks(rotation=45)
plt.show()
