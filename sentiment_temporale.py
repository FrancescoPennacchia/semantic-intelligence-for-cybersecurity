import pandas as pd
import matplotlib.pyplot as plt

# Leggi il file CSV
df = pd.read_csv('dati/dataseset_unitedkingdom_con_sentiment.csv', header=None, skiprows=1)

# Converti la colonna 'Data' in tipo datetime
df['Data'] = pd.to_datetime(df[2])

# Estrai il mese dalla colonna 'Data'
df['Mese'] = df['Data'].dt.to_period('M')

# Estrai il valore 'compound' dalla colonna 'Sentiment' (che è in formato JSON)
df['Compound'] = df[3].apply(lambda x: eval(x)['compound'])

# Raggruppa per mese e calcola la media del compund
media_mensile = df.groupby('Mese')['Compound'].mean()

# Crea il grafico
plt.figure(figsize=(12, 6))
media_mensile.plot(kind='line')
plt.axhline(y=0.05, color='red', linestyle='--', label='0.05')
plt.axhline(y=-0.05, color='red', linestyle='--', label='-0.05')
plt.title("Andamento del sentiment negli anni")
plt.xlabel('Anno')
plt.ylabel('Sentiment Medio')
plt.grid(True)

#plt.yticks([-0.1, -0.05, 0, 0.05, 0.1])
# Mostra il grafico
plt.show()
