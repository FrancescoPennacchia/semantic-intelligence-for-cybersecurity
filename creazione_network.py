import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd

# Caricamento dei dati
df = pd.read_csv('dati/dataseset_italia_con_sentiment.csv', header=None, skiprows=1)

# Assumi che la colonna 0 sia il testo e la colonna 4 sia il sentiment
texts = df[0].astype(str)
sentiments = df[4]

# Lista unica delle parole chiave
#keywords_list = ['euthanasia', 'people', 'think', 'know', 'right', 'life', 'years', 'person', 'law', 'case', 'referendum', 'Italy', 'get', 'end', 'many', 'someone', 'fact', 'suicide', 'Italian', 'problem', 'opinion', 'little', 'good', 'last', 'better', 'give', 'able', 'come', 'day', 'signatures', 'death', 'since', 'use', 'said', 'made', 'everyone', 'us', 'die', 'sense', 'ones', 'nothing', 'cant', 'hope', 'makes', 'legal', 'suffering']

#Pulite da chat gpt italia
keywords_list =[
    'euthanasia', 'life', 'death', 'law', 'referendum', 'Italy', 'Italian',
    'suicide', 'opinion', 'legal', 'suffering', 'Court', 'article', 'favor',
    'Cappato', 'civil', 'Euthanasia', 'choice', 'country', 'topic', 'vote',
    'rights', 'support', 'ill', 'referendums', 'society', 'murder',
    'Constitutional', 'information', 'human', 'issue', 'practice', 'political',
    'terminally', 'laws', 'doctors', 'doctor', 'consenting', 'consent', 'patient', 'moral', 'crime', 'abortion', 'God'
]


'''
Italia:
keywords_list = [
    'euthanasia', 'life', 'death', 'law', 'referendum', 'Italy', 'Italian',
    'suicide', 'opinion', 'legal', 'suffering', 'Court', 'article', 'favor',
    'Cappato', 'civil', 'Euthanasia', 'choice', 'country', 'topic', 'vote',
    'rights', 'support', 'ill', 'referendums', 'society', 'murder',
    'Constitutional', 'information', 'human', 'issue', 'practice', 'political',
    'terminally', 'laws', 'doctors', 'doctor', 'consenting', 'consent', 'patient', 'moral', 'crime', 'abortion'
]
'''
'''
francia:

[
    'euthanasia', 'death', 'life', 'law', 'suffering', 'suicide', 'opinion',
    'care', 'question', 'case', 'die', 'person', 'understand', 'subject',
    'longer', 'need', 'French', 'live', 'possible', 'debate', 'vote',
    'Euthanasia', 'family', 'country', 'suicide', 'law', 'something', 'debate',
    'vote', 'Euthanasia', 'family', 'animals', 'social', 'choice', 'euthanized',
    'patients', 'doctors', 'society', 'dignity', 'age', 'loved', 'Church',
    'religion', 'legal', 'support', 'health', 'pain', 'illness', 'assisted'
]
'''

'''
spagna
keywords_euthanasia = [
    'euthanasia', 'life', 'death', 'law', 'suicide', 'suffering', 'vote',
    'case', 'power', 'social', 'society', 'personal', 'questions', 'free',
    'die', 'opinion', 'years', 'action', 'person', 'hope', 'rights',
    'health', 'Euthanasia', 'pain', 'state', 'laws', 'decision', 'mental',
    'politics', 'education', 'dignified', 'government', 'patient', 'support',
    'countries', 'Podemos', 'progress', 'terminal', 'doctors'
]
'''

'''
Regno unito:

keywords_euthanasia = [
    'life', 'die', 'euthanasia', 'death', 'care', 'suicide', 'health',
    'person', 'country', 'government', 'suffering', 'family', 'pain',
    'issue', 'mental', 'legal', 'living', 'law', 'NHS', 'social', 'assisted',
    'support', 'state', 'free', 'terminal', 'ill', 'society', 'choice',
    'medical', 'doctors', 'patients', 'option', 'quality', 'human',
    'Euthanasia', 'doctor', 'decision', 'religious', 'terminally', 'treatment',
    'patient', 'illness', 'cases', 'court', 'laws', 'dignity'
]


'''

# Contare la frequenza del sentiment per ogni parola chiave
keyword_sentiment_count = {keyword: {'positive': 0, 'negative': 0, 'neutral': 0} for keyword in keywords_list}

for text, sentiment in zip(texts, sentiments):
    keywords_in_text = [keyword for keyword in keywords_list if keyword in text]
    for keyword in keywords_in_text:
        keyword_sentiment_count[keyword][sentiment] += 1

# Determinare il sentiment predominante per ogni parola chiave
keyword_dominant_sentiment = {}
for keyword, sentiments_count in keyword_sentiment_count.items():
    keyword_dominant_sentiment[keyword] = max(sentiments_count, key=sentiments_count.get)

# Creazione del grafo
G = nx.Graph()

# Aggiunta dei nodi al grafo con l'attributo sentiment predominante
for keyword, dominant_sentiment in keyword_dominant_sentiment.items():
    G.add_node(keyword, dominant_sentiment=dominant_sentiment)

# Aggiunta degli archi in base alle co-occorrenze
for text in texts:
    keywords_in_text = [keyword for keyword in keywords_list if keyword in text]
    for edge in combinations(keywords_in_text, 2):
        if G.has_edge(*edge):
            G[edge[0]][edge[1]]['weight'] += 1
        else:
            G.add_edge(*edge, weight=1)

# Calcolo delle misure di centralità e identificazione delle componenti connesse
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

# Identificazione delle componenti connesse (sotto-reti)
connected_components = list(nx.connected_components(G))

# Visualizzazione del grafo con i sentiment predominanti
pos = nx.spring_layout(G)
edge_labels = {(node1, node2): weight['weight'] for node1, node2, weight in G.edges(data=True)}
edge_widths = [weight['weight'] for _, _, weight in G.edges(data=True)]

sentiment_color_map = {'positive': 'green', 'neutral': 'blue', 'negative': 'red'}
node_colors = [sentiment_color_map[G.nodes[node]['dominant_sentiment']] for node in G.nodes()]

plt.figure(figsize=(12, 10))
nx.draw(G, pos, with_labels=True, font_size=10, font_color='black', node_size=800, edge_color='gray', alpha=0.9, node_color=node_colors, font_weight='bold') #, width=edge_widths
#nx.draw_networkx_edge_labels(G, pos, font_color='red') #, edge_labels=edge_labels
plt.title("Grafo con Co-occorrenza delle Parole Chiave e Sentiment Dominante")
plt.show()


print("degree_centrality: " + str(degree_centrality))
print("betweenness_centrality" + str(betweenness_centrality))
print("closeness_centrality" + str(closeness_centrality))
