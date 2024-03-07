import os
import string
from statistics import mean
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.corpus import stopwords

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


custom_words_tbbt=[
    'know','oh','okay','dont','youre','hey','well','want','thats','right','come','cant','shes',
    'sheldon', 'leonard', 'penny', 'howard', 'raj', 'bernadette', 'amy', 'lesley',
    'beverley', 'ramona', 'alicia', 'steph', 'kripke', 'yes', 'no', 'david', 'stephanie',
    'gablehouser', 'cooper', 'like', 'hee', 'leslie', 'missy', 'need', 'let', 'take',
    'would', 'could', 'come', 'bring', 'find', 'elizabeth', 'stan', 'zack', 'lee',
    'see', 'priya', 'mr', '...', 'make', 'hey', 'really', 'mean', 'one', 'jimmy',
    'yeah', 'okay', 'stuart', 'well', 'look', 'right', 'say', 'tell', 'go', 'get',
    'know', 'lucy', 'alex', 'oontz','arthur', 'janine', 'wolowitz', 'hello', 'wil',
    'dan', 'lorvis', 'emily', 'hofstadter', 'kevin', 'come', 'josh', 'amelia', 'abbott',
    'listen', 'roger', 'mary', 'claire', 'give', 'ask', 'alfred', 'play', 'call', 'bert',
    'isabella', 'susan', 'stop', 'good', 'beverley', 'koothrappali', 'lalita', 'bye', 'talk',
    'michaela', 'tom', 'wyatt', 'eat', 'glenn', 'two', 'try', 'open', 'christie', 'shelly',
    'like','back','going','yeah']






#Funzione che presi in ingresso i tre personaggi da analizzare e il corpus della serie,
#itera all'interno del corpus per estrarre, per ciascun personaggio solo i suoi dialoghi
# per ogni episodio di ogni stagione e crea un nuovo corpus in cui inserisce per ogni episodio
#e ogni stagione i dialoghi estratti.

def create_character_corpus(character_1, character_2, character_3, corpus):
    # Directory principale del corpus
    output_directory = f'{corpus} personaggi'
    os.makedirs(output_directory, exist_ok=True)
    characters = [character_1, character_2, character_3]
    for character in characters:

        character_directory = os.path.join(output_directory, character)
        os.makedirs(character_directory, exist_ok=True)

        for season_directory in os.listdir(corpus):
            season_path = os.path.join(corpus, season_directory)
            if os.path.isdir(season_path):

                character_season_directory = os.path.join(character_directory, season_directory)
                os.makedirs(character_season_directory, exist_ok=True)

                for episode_file in os.listdir(season_path):
                    episode_path = os.path.join(season_path, episode_file)
                    if os.path.isfile(episode_path):
                        with open(episode_path, 'r', encoding='utf-8') as file:
                            text = file.read()
                            dialogues_episode = [row.split(':')[1].strip() for row in text.lower().split('\n') if
                                                    ':' in row and character.lower() in row.split(':')[0]]

                            with open(os.path.join(character_season_directory, episode_file), 'w',
                                      encoding='utf-8') as character_file:
                                for dialogue in dialogues_episode:
                                    character_file.write(dialogue + '\n')

    return "Done!"


#Funzione che prende in input il path dell'episodio e ritorna una lista contenente
# tante stringhe, rese minuscole, quante sono le battute ripulite da punteggiatura,
# stop_words, parole con meno di tre caratteri, custom_words,
# caratteri numerici.
# Le stringhe vuote vengono eliminate, e lemmatizza.
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
        clean_text = [''.join([char for char in row if char not in string.punctuation]) for row in text]
        clean_text = [' '.join([word for word in row.split() if word not in english_stopwords]) for row in clean_text]
        clean_text = [''.join(char for char in battuta if not char.isdigit()) for battuta in clean_text]
        clean_text= [' '.join([word for word in row.split() if len(word) > 3]) for row in clean_text]
        clean_text = [' '.join([word for word in row.split() if word not in custom_words]) for row in clean_text]
        clean_text = [row for row in clean_text if row.strip()]

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
#dell'intero episodio, calcolando la media.
def sentiment_for_episode_vader(cleaned_text):
    analyzer = SentimentIntensityAnalyzer()
    sentiments_scores = [analyzer.polarity_scores(battuta)['compound'] for battuta in cleaned_text]

    if not sentiments_scores: # Gestione del caso in cui non ci siano punteggi da calcolare
        return 0.0

    vader_mean = mean(sentiments_scores)
    return vader_mean

#Funzione che prende in ingresso il testo restituito dalla funzione clean(episode) e calcola
#il sentiment utilizzando textblob, calcolando il sentiment di ogni battuta e succesivamente
#dell'intero episodio, calcolando la media.

def sentiment_for_episode_textblob(cleaned_text):
    sentiment_scores = [TextBlob(phrase).sentiment.polarity for phrase in cleaned_text]

    if not sentiment_scores:
        return 0.0

    textblob_mean = mean(sentiment_scores)
    return textblob_mean

def character_episodes_sentiment(root_dir, characters,custom_words,serie_name):
    dataframe_dir = 'df'
    os.makedirs(dataframe_dir, exist_ok=True)
    data = []

    for character in characters:
        character_path = os.path.join(root_dir, character)
        if not os.path.isdir(character_path):
            print(f'No data available for {character}.')
            continue

        for season_dir in sorted(os.listdir(character_path)):
            season_path = os.path.join(character_path, season_dir)
            if not os.path.isdir(season_path):
                continue

            episodes_data = []

            for episode_file in sorted(os.listdir(season_path)):
                if episode_file.endswith('.txt'):
                    episode_path = os.path.join(season_path, episode_file)
                    clean_text=clean(episode_path,custom_words)
                    vader_sentiment_label=assign_sentiment_label(sentiment_for_episode_vader(clean_text))
                    vader_sentiment_numeric=sentiment_for_episode_vader(clean_text)
                    textblob_sentiment_label=assign_sentiment_label(sentiment_for_episode_textblob(clean_text))
                    textblob_sentiment_numeric=sentiment_for_episode_textblob(clean_text)

                    episodes_data.append([character, season_dir, episode_file, vader_sentiment_label, textblob_sentiment_label,round(vader_sentiment_numeric,4),round(textblob_sentiment_numeric,4)])
            # Ordina gli episodi in base al numero dell'episodio
            episodes_data = sorted(episodes_data, key=lambda x: int(x[2].split('_')[1].split('.')[0]))

            data.extend(episodes_data)

    df = pd.DataFrame(data, columns=['Character', 'Season', 'Episode', 'Vader_sent_label', 'Textblob_sent_label','Vader_sent_numeric','Textblob_sent_numeric'])
    output_csv_path = os.path.join(dataframe_dir, f'3_episode_results_{serie_name}.csv')
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    return df



def character_season_sentiment(root_dir,characters,custom_words,serie_name):
    dataframe_dir = 'df'
    os.makedirs(dataframe_dir, exist_ok=True)
    data = []

    for character in characters:
        character_path = os.path.join(root_dir, character)
        if not os.path.isdir(character_path):
            print(f'No data available for {character}.')
            continue

        for season_dir in sorted(os.listdir(character_path)):
            season_path = os.path.join(character_path, season_dir)
            if not os.path.isdir(season_path):
                continue
            vader_sentiments = []
            textblob_sentiments = []
            for episode_file in sorted(os.listdir(season_path)):
                if episode_file.endswith('.txt'):
                    episode_path = os.path.join(season_path, episode_file)
                    clean_text = clean(episode_path,custom_words)
                    vader_sentiment = sentiment_for_episode_vader(clean_text)
                    vader_sentiments.append(vader_sentiment)
                    textblob_sentiment = sentiment_for_episode_textblob(clean_text)
                    textblob_sentiments.append(textblob_sentiment)

            avg_vader_sentiment = sum(vader_sentiments) / len(vader_sentiments) if len(vader_sentiments) > 0 else 0.0
            avg_textblob_sentiment = sum(textblob_sentiments) / len(textblob_sentiments) if len(
                textblob_sentiments) > 0 else 0.0


            data.append([character, season_dir, assign_sentiment_label(avg_vader_sentiment), assign_sentiment_label(avg_textblob_sentiment),round(avg_vader_sentiment,4),round(avg_textblob_sentiment,4)])
    df = pd.DataFrame(data, columns=['Character', 'Season', 'Vader_sent_label', 'Textblob_sent_label','Vader_sent_numeric','Textblob_sent_numeric'])
    df.to_csv(os.path.join(dataframe_dir, f'3_season_results_{serie_name}.csv'), index=False,
                      encoding='utf-8')

    return df




if __name__=='__main__':
    #create_character_corpus('damon','stefan','elena','corpus the vampire diaries')
    #create_character_corpus('sheldon','leonard','penny','corpus the big bang theory')
    #print(character_episodes_sentiment('corpus the vampire diaries personaggi',['stefan', 'damon', 'elena'],custom_words_tvd,'tvd'))
    #print(character_season_sentiment('corpus the vampire diaries personaggi',['stefan', 'damon', 'elena'],custom_words_tvd,'tvd'))
    #print(character_episodes_sentiment('corpus the big bang theory personaggi',['sheldon', 'leonard', 'penny'],custom_words_tbbt,'tbbt'))
    #print(character_season_sentiment('corpus the big bang theory personaggi',['sheldon', 'leonard', 'penny'],custom_words_tbbt,'tbbt'))
