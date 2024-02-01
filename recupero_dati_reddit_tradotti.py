import praw
import json
from datetime import datetime
from deep_translator import GoogleTranslator

def traduci_testo(testo):
    traduttore = GoogleTranslator(source='en', target='en')
    # Prendi solo i primi 5000 caratteri
    testo_ridotto = testo[:4999]
    return traduttore.translate(testo_ridotto)

# Credenziali dell'app Reddit
client_id = 'I3eXFCdwdiJlMzhbln3kRg'
client_secret = 'XRqs1RvT-DvFbzsre8Ywm2oPLC9P-Q'
username = 'FraReturn'
user_agent = 'user_agent'

# Inizializza l'oggetto PRAW Reddit
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     username=username,
                     user_agent=user_agent)
#SpainPolitics
#italy
# Parametri di ricerca
subreddit_name = 'unitedkingdom'
search_query = 'Euthanasia'
num_posts = 50  # Numero di post da recuperare




# Esegui la ricerca nel subreddit
subreddit = reddit.subreddit(subreddit_name)
posts = subreddit.search(search_query, limit=num_posts)

# Creazione di un dataset
data = []

num = 0
for post in posts:
    print(num)
    num += 1
    post_data = {
        'Titolo': traduci_testo(post.title),
        'Testo del post': traduci_testo(post.selftext),
        'Data': datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
        'Commenti': []
    }
     #data.append(post_data)

    for comment in post.comments:
        if isinstance(comment, praw.models.MoreComments):
            continue  # Salta gli oggetti MoreComments

        if "[deleted]" or "[removed]" not in comment.body:
            comment_data = {
                'Testo del commento': traduci_testo(comment.body),
                'Data del commento': datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                'Risposte': [],
            }

            # Aggiungi le risposte ai commenti
            for reply in comment.replies:
                if isinstance(reply, praw.models.MoreComments):
                    continue  # Salta gli oggetti MoreComments
                if "[deleted]" or "[removed]" not in reply.body:
                    reply_data = {
                        'Testo della risposta': traduci_testo(reply.body),
                        'Data della risposta': datetime.utcfromtimestamp(reply.created_utc).strftime(
                            '%Y-%m-%d %H:%M:%S'),
                    }
                    comment_data['Risposte'].append(reply_data)
            post_data['Commenti'].append(comment_data)
    data.append(post_data)

# Salvataggio della lista di dizionari in un file JSON
with open('dati/dati_sub_reddit_unitedkingdom.json', 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)