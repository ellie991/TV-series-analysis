import os
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
from statistics import mean


english_stopwords=stopwords.words('english')


custom_words_tvd=[
    'know','oh','okay','dont','youre','hey','well','want','thats','right','come','cant','shes','damon',
    'elena','stefan','caroline','klaus', 'jeremy','bonnie','katherine','silas',
    'elijah','enzo', 'lexi', 'tyler','kai','alaric', 'rebekah','anna','luke',
    'jenna','hayley','mason','sybil','vicki','jo','john','mickael','qetsiyah','lillian',
    'liv','kol','mikaelson','isobel','nora','april','liz','luka','mary','emily','zach',
    'julian','nadia','cade','logan','seline','connor','markos','bill','jules',
    'andie','esther','lucy','ben','matt','aaron','finn','darren','james',
    'well', 'right', 'okay','frederick','josie','helen','louise','noah','sarah','liam','meredith',
    'sandwich','like','back','going','yeah']

custom_words_tbbt=['know','oh','okay','dont','youre','hey','well','want','thats','right','come','cant','shes','damon','elena','stefan','caroline','klaus', 'jeremy','bonnie','katherine','silas',
 'elijah','enzo', 'lexi', 'tyler','kai','alaric', 'rebekah','anna','luke',
 'jenna','hayley','mason','sybil','vicki','jo','john','mickael','qetsiyah','lillian',
 'liv','kol','mikaelson','isobel','nora','april','liz','luka','mary','emily','zach',
 'julian','nadia','cade','logan','seline','connor','markos','bill','jules',
 'andie','esther','lucy','ben','matt','aaron','finn','darren','james',
'well', 'right', 'okay','frederick','josie','helen','louise','noah','sarah','liam','meredith',
'sandwich','like','back','going','yeah']

#Funzione per la pulizia del testo che prende in input il path dell'episodio e ritorna una
# lista contenente tante stringhe, rese minuscole, quante sono le battute ripulite da punteggiatura,
# stop_words, parole con meno di tre caratteri, custom_words,
# caratteri numerici. Viene utilizzata anche la lemmatizzazione.
# Le stringhe vuote vengono eliminate.

def clean(episode,custom_words):
    with open(episode, 'r', encoding='utf-8') as episode_file:
        episode_text = episode_file.read().split('\n')
        text = []
        for row in episode_text:
            row = row.lower()
            if ':' in row:
                parts = row.split(':')
                text.append(parts[1])
            else:
                text.append(row)
        clean_text = [''.join([char for char in battuta if char not in string.punctuation]) for battuta in text]
        clean_text = [' '.join([word for word in row.split() if word not in english_stopwords]) for row in clean_text]
        clean_text = [''.join(char for char in battuta if not char.isdigit()) for battuta in clean_text]
        clean_text = [' '.join([word for word in row.split() if len(word) > 3]) for row in clean_text]
        clean_text = [' '.join([word for word in row.split() if word not in custom_words]) for row in clean_text]
        clean_text = [row for row in clean_text if row.strip()]
        clean_text = [' '.join([word for word in row.split()]) for row in clean_text]

    return clean_text




#Funzione che preso in ingresso un valore numerico, restituisce una stringa in base
# al valore che questo assume.
def assign_sentiment_label(value):
    if value > 0:
        return 'positive'
    elif value < 0:
        return 'negative'
    else:
        return 'neutral'




#Funzione che prende in ingresso il testo restituito dalla funzione clean(episode) e calcola
#il sentiment utilizzando vader, calcolando il sentiment di ogni battuta e succesivamente
#dell'intero episodio, restituendo la media.
def sentiment_for_episode_vader(clean_text):
    analyzer = SentimentIntensityAnalyzer()
    sentiments_scores=[analyzer.polarity_scores(battuta)['compound'] for battuta in clean_text]
    vader_mean=mean(sentiments_scores)
    return vader_mean


#Funzione che prende in ingresso il testo restituito dalla funzione clean(episode) e calcola
#il sentiment utilizzando textblob, calcolando il sentiment di ogni battuta e succesivamente
#dell'intero episodio, restituendo la media.

def sentiment_for_episode_textblob(clean_text):
    sentiment_scores = [TextBlob(phrase).sentiment.polarity for phrase in clean_text]
    textblob_mean = mean(sentiment_scores)
    return textblob_mean



#Funzione che calcola il valore del sentiment per ogni episodio di ogni stagione, utilizzando
#sia vader che textblob e salva tutti i risultati in un dataframe, che inserisce in
#un'apposita directory, che se non esiste crea, memorizzando i valori di sentiment sia
#in forma numerica che categorica.


def visualize_episodes_sentiment(root_dir,serie_name,custom_words):
    dataframe_dir = 'df'
    os.makedirs(dataframe_dir, exist_ok=True)
    data=[]
    for season_dir in sorted(os.listdir(root_dir)):
        season_path = os.path.join(root_dir,season_dir)
        episodes_data = []

        for episode_file in sorted(os.listdir(season_path)):
            if episode_file.endswith('.txt'):
                episode_path = os.path.join(season_path, episode_file)
                clean_text = clean(episode_path, custom_words)
                vader_sentiment_label= assign_sentiment_label(sentiment_for_episode_vader(clean_text))
                textblob_sentiment_label=assign_sentiment_label(sentiment_for_episode_textblob(clean_text))
                vader_sentiment_numeric=sentiment_for_episode_vader(clean_text)
                textblob_sentiment_numeric=sentiment_for_episode_textblob(clean_text)
                episodes_data.append([season_dir, episode_file, vader_sentiment_label, textblob_sentiment_label,round(vader_sentiment_numeric,4),round(textblob_sentiment_numeric,4)])

            #Ordina gli episodi in base al numero dell'episodio

            episodes_data= sorted(episodes_data, key=lambda x: int(x[1].split('_')[1].split('.')[0]))

        data.extend(episodes_data)


    df= pd.DataFrame(data, columns=['Season', 'Episode', 'Vader_sent_label', 'Textblob_sent_label','Vader_sent_numeric','Textblob_sent_numeric'])
    output_csv_path = os.path.join(dataframe_dir, f'2_episode_results_{serie_name}.csv')
    df.to_csv(output_csv_path, index=False, encoding='utf-8')

    return df

#Funzione che calcola il valore del sentiment per ogni stagione, utilizzando
#sia vader che textblob e salva tutti i risultati in un dataframe, che inserisce in
#un'apposita directory, che se non esiste crea, memorizzando i valori di sentiment sia
#in forma numerica che categorica.

def visualize_season_sentiment(root_dir,serie_name,custom_words):
    dataframe_dir = 'df'
    os.makedirs(dataframe_dir, exist_ok=True)
    data = []

    for season_dir in sorted(os.listdir(root_dir)):
        season_path = os.path.join(root_dir, season_dir)
        if os.path.isdir(season_path):
            vader_sentiments = []
            textblob_sentiments = []

            for episode_file in sorted(os.listdir(season_path)):
                if episode_file.endswith('.txt'):
                    episode_path = os.path.join(season_path, episode_file)
                    clean_text = clean(episode_path, custom_words)

                    vader_sentiment = sentiment_for_episode_vader(clean_text)
                    vader_sentiments.append(vader_sentiment)

                    textblob_sentiment = sentiment_for_episode_textblob(clean_text)
                    textblob_sentiments.append(textblob_sentiment)

            avg_vader_sentiment = mean(vader_sentiments) if len(vader_sentiments) > 0 else 0.0
            avg_textblob_sentiment = mean(textblob_sentiments) if len(textblob_sentiments) > 0 else 0.0
            data.append([season_dir, assign_sentiment_label(avg_vader_sentiment), assign_sentiment_label(avg_textblob_sentiment),round(avg_vader_sentiment,4),round(avg_textblob_sentiment,4)])

    df = pd.DataFrame(data, columns=['Season', 'Vader_sent_label', 'Textblob_sent_label','Vader_sent_numeric','Textblob_sent_numeric'])
    df.to_csv(os.path.join(dataframe_dir, f'2_season_results_{serie_name}.csv'), index=False, encoding='utf-8')

    return df



if __name__=='__main__':
    #print(visualize_episodes_sentiment('corpus the vampire diaries','tvd',custom_words_tvd))
    #print(visualize_season_sentiment('corpus the vampire diaries','tvd',custom_words_tvd))
    print(visualize_episodes_sentiment('corpus the big bang theory','tbbt',custom_words_tbbt))
    print(visualize_season_sentiment('corpus the big bang theory','tbbt',custom_words_tbbt))


