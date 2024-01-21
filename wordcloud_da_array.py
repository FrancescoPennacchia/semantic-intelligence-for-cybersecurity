from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Example words and their sentiment values
words = ['would', 'dont', 'like', 'even', 'one', 'also', 'euthanasia', 'people', 'think', 'Im', 'know', 'right', 'want', 'life', 'time', 'make', 'years', 'already', 'case', 'person', 'law', 'see', 'say', 'go', 'understand', 'much', 'could', 'referendum', 'without', 'doesnt', 'things', 'way', 'still', 'take', 'Italy', 'always', 'get', 'end', 'must', 'fact', 'many', 'someone', 'well', 'possible', 'something', 'suicide', 'two', 'Italian', 'least', 'problem', 'sign', 'opinion', 'didnt', 'thing', 'really', 'find', 'little', 'point', 'anyone', 'cases', 'live', 'good', 'last', 'never', 'first', 'need', 'new', 'better', 'long', 'able', 'give', 'others', 'another', 'come', 'said', 'signatures', 'day', 'use', 'seems', 'death', 'made', 'since', 'die', 'us', 'legal', 'certain', 'everyone', 'ones', 'nothing', 'remember', 'sense', 'hope', 'cant', 'less', 'work', 'question', 'going', 'abortion', 'makes', 'However', 'therefore', 'put', 'every', 'Ill', 'read', 'suffering', 'everything', 'given', 'today', '2', 'done', 'Court', 'believe', 'article', 'favor', 'longer', 'ago', 'feel', 'anything', 'ask', 'due', 'serious', 'etc', 'lot', 'almost', 'Cappato', 'away', 'civil', 'situation', 'help', 'else', 'family', 'start', 'example', 'Euthanasia', 'days', '3', 'change', 'different', 'talk', 'video', 'cycle', 'choice', 'look', 'country', 'topic', 'seem', 'car', 'Well', 'rather', 'completely', 'part', 'yes', 'certainly', 'tell', 'obviously', 'ever', 'state', 'back', 'pass', 'problems', 'full', 'try', 'city', 'old', 'place', 'bad', 'perhaps', 'vote', 'used', 'next', 'possibility', 'reasons', 'means', 'thought', 'crime', 'wants', 'world', 'times', 'rights', 'months', 'around', 'talking', 'free', 'year', 'let', 'soon', 'friend', 'support', 'Lets', 'may', 'ill', 'actually', 'mean', 'shit', 'instead', 'simply', 'referendums', 'course', 'child', 'clear', 'wrong', 'decide', 'society', 'murder', 'Constitutional', 'paths', 'youre', 'post', 'Ive', 'important', 'choose', 'information', 'enough', '1', 'making', 'sentence', 'agree', 'probably', 'wanted', 'maybe', 'mine', 'care', 'isnt', 'moment', 'best', 'thinking', 'simple', 'reason', 'level', 'keep', 'view', 'went', 'left', 'idea', 'morning', 'girl', 'call', 'comment', 'social', 'towards', 'discussion', 'easy', 'bring', 'news', 'road', 'happens', 'called', 'story', 'personal', 'says', 'suffer', 'answer', 'human', 'issue', 'seen', 'otherwise', 'found', 'saying', 'however', 'difficult', 'age', 'practice', 'political', 'interested', 'reality', 'especially', 'cars', 'woman', 'collection', 'request', 'Thanks', 'fuck', 'online', 'anyway', 'terminally', 'Yes', 'nice', 'kill', 'repeal', 'Even', 'lets', 'laws', 'doctors', 'often', 'whether', 'short', 'prison', 'cannabis', 'friends', 'sure', 'become', 'doctor', 'type', 'leave', 'stop', 'consenting', 'consent', 'money', 'various', 'patient', 'thats', 'decided', 'system', 'knows', 'Maybe', 'happened', 'birth', 'bike', 'freedom', 'economic', 'matter', 'Oh', 'pain', 'absolutely', 'wait', 'side', 'commit', 'alone', 'mental', 'yet', 'light', 'living', 'consider', 'move', 'avoid', 'body', 'assisted', 'stuff', 'impossible', 'comes', 'area', 'wrote', 'home', 'bit', 'love', 'mother', 'experience', 'yesterday', '100', 'small', 'similar', 'hours', 'number', 'risk', 'pay', 'sorry', 'open', '20', 'children', 'far', 'mind', 'man', 'Switzerland', 'lives', 'future', 'force', 'wont', 'couple', 'parliament', 'party', 'Unfortunately', 'second', 'half', 'damn', 'needs', 'interesting', 'got', 'goes', 'signature', 'directly', 'rest', 'real', 'Milan', 'remain', 'words', 'written', 'art', '10', 'comments', 'exactly', 'explain', 'Id', 'happy', 'started', 'period', 'Hi', 'wine', 'within', 'took', 'beautiful', 'later', 'among', 'general', 'thousand', 'send', 'allow', 'cities', 'Obviously', 'young', 'Catholic', 'according', 'Good', 'text', 'apply', 'wasnt', 'immediately', 'medical', 'patients', 'hand', 'though', 'kind', 'happen', 'majority', 'healthy', 'normal', 'path', 'buy', 'past', 'night', 'minutes', 'thanks', 'valid', 'respect', 'signed', 'boy', 'legalization', 'born', 'imagine', 'wouldnt', 'weeks', 'usual', 'behind', 'practically', 'clearly', 'considered', 'truly', 'cause', 'absurd', 'taking', '50', 'purpose', 'looking', 'politics', 'English', 'recommend', 'dangerous', 'big', 'week', 'Rome', 'using', 'cyclists', 'public', 'house', 'died', 'andor', 'association', 'changed', 'quality', 'parties', 'individual', 'write', 'knowing', 'worth', 'conditions', 'consequences', 'radicals', '5', 'saw', 'seeing', 'add', 'except', 'women', 'three', 'spend', 'worse', 'ruling', 'price', 'poor', 'Christmas', 'passed', 'great', 'told', 'usually', 'collect', 'God', 'situations', 'depression', 'Luca', 'Coscioni', 'necessary', 'prevent', 'line', 'religious', 'common', 'office', 'questions', 'code', 'result', 'prefer']


def sentiment(word):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(word)

    if sentiment_scores['compound'] >= 0.05:
        return 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# First, we will apply the 'sentiment' function to each word in the list 'words'
word_sentiments = {word: sentiment(word) for word in words}

# Then, we will filter the words to include only those with positive sentiment
positive_words_only = {word for word, sentiment in word_sentiments.items() if sentiment == 'positive'} # Change 'positive' to 'negative' or 'neutral


# Calcolare le frequenze delle parole positive
positive_word_frequencies = {word: words.count(word) for word in positive_words_only}

# Generare la wordcloud per le parole con sentiment positivo
positive_wordcloud = WordCloud(width=800, height=800,
                               background_color='white',
                               min_font_size=10).generate_from_frequencies(positive_word_frequencies)

# Visualizzare la wordcloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(positive_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()
