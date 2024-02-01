import nltk
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

dataset = pd.read_csv('dati/dataseset_unitedkingdom_con_sentiment.csv', header=None, skiprows=1)

''' #Pulite da chat gpt italia
keywords = [
    'euthanasia', 'life', 'death', 'law', 'referendum', 'Italy', 'Italian',
    'suicide', 'opinion', 'legal', 'suffering', 'Court', 'article', 'favor',
    'Cappato', 'civil', 'Euthanasia', 'choice', 'country', 'topic', 'vote',
    'rights', 'support', 'ill', 'referendums', 'society', 'murder',
    'Constitutional', 'information', 'human', 'issue', 'practice', 'political',
    'terminally', 'laws', 'doctors', 'doctor', 'consenting', 'consent', 'patient'
] '''

''' #Spagna
keywords = [
    'euthanasia', 'death', 'life', 'law', 'suffering', 'suicide', 'opinion',
    'care', 'question', 'case', 'die', 'person', 'understand', 'subject',
    'longer', 'need', 'French', 'live', 'possible', 'debate', 'vote',
    'Euthanasia', 'family', 'country', 'suicide', 'law', 'something', 'debate',
    'vote', 'Euthanasia', 'family', 'animals', 'social', 'choice', 'euthanized',
    'patients', 'doctors', 'society', 'dignity', 'age', 'loved', 'Church',
    'religion', 'legal', 'support', 'health', 'pain', 'illness', 'assisted'
] '''

# Estrazione delle frasi dalla colonna "Frase"
all_frasi = "".join(dataset[0].astype(str))

tokens = [word.lower() for word in nltk.word_tokenize(all_frasi) if word.lower() not in stopwords.words("english") and word.isalnum()]

# Conteggio delle frequenze delle parole chiave
word_counts = pd.DataFrame.from_dict(Counter(tokens),orient='index',columns=['count']).reset_index().rename(columns={'index':'word'})

# Ordina le parole chiave in base alla frequenza (dalla più alta alla più bassa)
top_words = word_counts.sort_values(by=['count'], ascending=False).head(20)

# Creazione del grafico a barre con Seaborn
sns.set_style("whitegrid")
plt.figure(figsize=(10,8))
sns.barplot(x="word", y="count", data=top_words, color="m")
plt.title("Top 20 parole più frequenti nei commenti del Regno Unito")
plt.xlabel("Parole")
plt.ylabel("Conteggio")
plt.show()
