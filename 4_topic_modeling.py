import os
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

names_tvd = ['damon', 'elena', 'stefan', 'caroline', 'klaus', 'jeremy', 'bonnie', 'katherine', 'silas',
             'elijah', 'enzo', 'lexi', 'tyler', 'kai', 'alaric', 'rebekah', 'anna', 'luke',
             'jenna', 'hayley', 'mason', 'sybil', 'vicki', 'jo', 'john', 'mickael', 'qetsiyah', 'lillian',
             'liv', 'kol', 'mikaelson', 'isobel', 'nora', 'april', 'liz', 'mary', 'emily', 'zach',
             'julian', 'nadia', 'cade', 'logan', 'seline', 'connor', 'markos', 'bill', 'jules',
             'andie', 'esther', 'lucy', 'ben', 'matt', 'aaron', 'finn', 'darren', 'james',
             'well', 'right', 'okay', 'frederick', 'josie', 'helen', 'louise', 'noah', 'sarah', 'liam', 'meredith',
             'sandwich', 'lily', 'abby', 'kelly', 'peter', 'exact', 'henry', 'oscar', 'cut', 'nice', 'think', 'want',
             'arrived', 'ouch', 'unless','joshua','sheila',
             'jane', 'kimberley', 'nope', 'alright', 'tessa', 'wait', 'whoa', 'valerie', 'malcolm',
             'minute', 'anne', 'literally', 'earlier', 'mikael', 'luka', 'thing', 'rayna',
             'virginia', 'voiceover', 'start', 'anne', 'rudy', 'yeah', 'mario', 'giuseppe', 'maggie', 'alex', 'penny',
             'marty', 'matty','trevor', 'megan','samantha','jamie','jesse','aimee']

names_tbbt = ['sheldon', 'leonard', 'penny', 'howard', 'raj', 'bernadette', 'amy', 'like',
              'yeah', 'okay', 'stuart', 'well', 'right', 'lesley', 'beverley', 'ramona', 'alicia',
              'steph', 'kripke', 'yes', 'no', 'david', 'stephanie', 'rajesh',
              'gablehouser', 'cooper', 'like', 'hee', 'leslie', 'missy', 'need', 'let', 'take',
              'would', 'could', 'come', 'bring', 'find', 'elizabeth', 'stan', 'zack', 'lee',
              'see', 'priya', 'make', 'hey', 'really', 'mean', 'one', 'jimmy', 'howie'
              'yeah', 'okay', 'stuart', 'well', 'look', 'right', 'say', 'tell', 'go', 'get',
              'know', 'lucy', 'alex', 'oontz', 'arthur', 'janine', 'wolowitz', 'hello', 'wil',
              'dan', 'lorvis', 'emily', 'hofstadter', 'kevin', 'come', 'josh', 'amelia', 'abbott',
              'listen', 'roger', 'mary', 'claire', 'give', 'ask', 'alfred', 'play', 'call', 'bert',
              'isabella', 'susan', 'stop', 'good', 'beverley', 'koothrappali', 'lalita', 'bye', 'talk',
              'michaela', 'tom', 'wyatt', 'eat', 'glenn', 'two', 'try', 'open', 'christie', 'shelly']

verbs = ['look', 'know', 'walk', 'like', 'take', 'come', 'need', 'going', 'go', 'make', 'turn', 'tell', 'would',
         'could', 'find', 'give','spell','speak',
         'looking', 'mean', 'walking', 'keep', 'talk', 'said', 'call', 'thought', 'trying', 'talking', 'sitting',
         'told','seems','stand','become','becomes',
         'answering','thinking', 'caused', 'putting', 'pulling', 'spoke', 'arrives', 'enters', 'seemed',
         'seen', 'pull', 'push', 'continues', 'washing','speaks','sits','turned','shall']


other_words=['stop', 'something', 'really', 'little', 'good', 'sorry',
               'sure', 'table', 'fine', 'first', 'melunaweh', 'beep', 'recent', 'phone',
               'mood', 'matter', 'thank', 'early','truly','otherwise','semi',
               'bathrobe', 'else', 'mail', 'brightly', 'bell', 'bathroom', 'entry', 'true',
               'item', 'back', 'away', 'around', 'still', 'behind', 'never', 'even', 'front', 'anything', 'thing',
               'last', 'everything', 'another', 'much','neither','whether','anyone','zero','throw'
               'maybe', 'next', 'nothing', 'someone', 'toward', 'suddenly', 'anybody', 'towards', 'since',
               'anyway', 'within', 'almost', 'wherever', 'throw', 'everywhere', 'highly', 'midly', 'sale',
                'eventually','except','ordinary', 'device','across',
               'ungh','maybe','aside','every','woah','goodbye', 'shoe','afternoon','anytime',
             'others', 'whenever']



# Funzione per la pulizia del testo, prende in ingresso il tsto di un episodio
#elimina il nome del personaggio all'inizio di ogni battuta.
#Elimina la punteggiatura,tokenizza, elmina le stopwords, lemmatizza e infine
#elimina le parole con meno di 4 caratteri, i caratteri numerici e tutte le parole che rientrano
#nelle precedenti liste.
def clean_text(text):
    text = text.split('\n')  # si splitta il testo dividendole in battute
    script = ''
    for row in text:
        if ':' in row:
            parts = row.split(':')
            script += parts[1]
        else:
            script += row
    script = re.sub(r'[^\w\s]', ' ', script)
    tokens = word_tokenize(script.lower())
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if len(token) > 3
              and token not in names_tvd and token not in names_tbbt and token not in verbs and
              token not in other_words and not token.isdigit()]
    tokens = ' '.join(tokens)  # unisce tutti i token in una stringa

    return tokens


# Funzione che prende in ingresso il corpus, per ottenere tutti gli script degli episodi.

def get_episode_scripts(corpus_directory):
    season_data = {}  # Crea un dizionario per inserire gli script delle stagioni

    for season in os.listdir(corpus_directory):
        season_path = os.path.join(corpus_directory, season)
        if os.path.isdir(season_path):
            season_data[season] = []
            # Inizializza una lista per inserire gli script della stagione
            for episode_file in os.listdir(season_path):

                episode_file_path = os.path.join(season_path, episode_file)
                if os.path.isfile(episode_file_path):
                    with open(episode_file_path, 'r', encoding='latin-1') as file:
                        script = file.read()
                        script = clean_text(script)
                        season_data[season].append(script) #Aggiunge gli script della stagione alla lista delle stagioni

    return season_data


# Funzione per estrarre 5 topic con il modello LDA
def topics_lda(seasons_scripts, top=5):
    tfidf = TfidfVectorizer(strip_accents='unicode',
                            max_df=0.99, min_df=2, max_features=5000)  # crea un oggetto TfidfVectorizer

    X_tfidf = tfidf.fit_transform(
        seasons_scripts)  # trasforma gli script della stagione in una rappresentazione vettoriale TF-IDF

    lda = LatentDirichletAllocation(n_components=top, learning_method='online', batch_size=256, max_iter=5,
                                    random_state=42)  # configura il modello LDA

    lda.fit(X_tfidf)  # addestra il modello LDA sugli script della stagione

    feature_names = tfidf.get_feature_names_out()  # estrae le parole chiave per ciascun topic
    topics = []

    for topic_index, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-top - 1:-1][:top]]
        top_words_weights = [round(topic[i], 5) for i in topic.argsort()[:-top - 1:-1][:top]]
        topics.append({
            'topic_index': topic_index,
            'top_words': top_words,
            'top_words_weights': top_words_weights}
        )

    return topics

#Funzione per estrarre 5 topic con il modello NMF
def topics_nmf(seasons_scripts, top=5):
    tfidf = TfidfVectorizer(strip_accents='unicode',
                            max_df=0.99, min_df=2, max_features=5000)  # crea un oggetto TfidfVectorizer

    X_tfidf = tfidf.fit_transform(
        seasons_scripts)  # trasforma gli script della stagione in una rappresentazione vettoriale TF-IDF

    nmf = NMF(n_components=top, max_iter=1000, random_state=42)

    nmf.fit(X_tfidf)  # addestra il modello NMF sugli script della stagione
    feature_names = tfidf.get_feature_names_out()  # estrae le parole chiave per ciascun topic
    topics = []

    for topic_index, topic in enumerate(nmf.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-top - 1:-1][:top]]
        top_words_weights = [round(topic[i],5) for i in topic.argsort()[:-top - 1:-1][:top]]  # Aggiungi i pesi TF-IDF
        topics.append({
            'topic_index': topic_index,
            'top_words': top_words,
            'top_words_weights': top_words_weights})

    return topics


# Funzione per eseguire la topic modeling separatamente per ogni stagione, restituendo
# 5 topic per ogni stagione, prende in ingresso gli script delle stagioni come una
# lista di liste e un metodo tra LDA e NMF
def print_season_topics(seasons_scripts, method):
    all_seasons_topics = {}  # inizializza un dizionario in cui salvare per ogni stagione i topic
    if method.lower() == 'lda':
        print('Topics LDA for seasons')

        for season in seasons_scripts:
            # itera attraverso gli script di ogni stagione e calcola i topic utilizzando LDA
            topics = topics_lda(seasons_scripts[season]) # calcola i topic con funzione lda creata
            all_seasons_topics[season] = topics # aggiunge i topic al dizionario


    elif method.lower() == 'nmf':
        print('Topics NMF for seasons')

        for season in seasons_scripts:
            topics = topics_nmf(seasons_scripts[season])
            all_seasons_topics[season] = topics
    else:
        print('Error: Invalid method')

    for season in all_seasons_topics:
        print(season)
        print(all_seasons_topics[season])

    return all_seasons_topics


# Funzione per eseguire la topic modeling sull'intera serie, restituendo 5 topic
# per l'intera serie
def print_global_topics_(seasons_scripts, method):
    global_season_data = []
    # Itera tra le chiavi del dizionario
    for season_key,season_scripts in seasons_scripts.items():
        # Unione di tutte le stringhe corrispondenti alla stagione e aggiungile alla lista globale
        global_season_text = ' '.join(season_scripts)
        global_season_data.append(global_season_text)

    if method.lower() == 'lda':
        print('Topics LDA for the series')
        topics = topics_lda(global_season_data)  # calcola i topic sull'intera serie utilizzando LDA

        for topic in topics:  # print dei topic ottenuti
            print(topic)
    elif method.lower() == 'nmf':
        print('Topics NMF for the series')

        topics = topics_nmf(global_season_data)

        for topic in topics:
            print(topic)
    else:
        print('Error: Invalid method')  # se il metodo inserito non corrisponde ne a LDA ne a NMF

    return topics




#Wordcloud per visualizzare i 5 topic globali si una serie.

def create_wordcloud(topics):
    plt.figure(figsize=(20, 10))  # Dimensioni della figura

    for i in range(5):
        current_topic = topics[i]
        terms = current_topic['top_words']
        tfidf_weights = current_topic['top_words_weights']
        tfidf_dict = {terms[j]: tfidf_weights[j] for j in range(len(terms))}
        wordcloud = WordCloud(width=400, height=200, background_color='white',
                              prefer_horizontal=1.0, relative_scaling=0.5)
        wordcloud.generate_from_frequencies(tfidf_dict)
        plt.subplot(1, 5, i + 1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

    plt.tight_layout()
    plt.show()



#Wordcloud per visualizzare in modo affiancato ciascun topic delle due serie.
def create_combined_wordclouds(topics1, topics2):
    for i in range(len(topics1)):
        words1 = topics1[i]['top_words']
        weights1 = topics1[i]['top_words_weights']
        words2 = topics2[i]['top_words']
        weights2 = topics2[i]['top_words_weights']
        tfidf_dict1 = {words1[j]: weights1[j] for j in range(len(words1))}
        tfidf_dict2 = {words2[j]: weights2[j] for j in range(len(words2))}
        wordcloud1 = WordCloud(width=800, height=400, background_color='white')
        wordcloud1.generate_from_frequencies(tfidf_dict1)

        wordcloud2 = WordCloud(width=800, height=400, background_color='white')
        wordcloud2.generate_from_frequencies(tfidf_dict2)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(wordcloud1, interpolation='bilinear')
        plt.title(f'Topic {i} - TVD')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(wordcloud2, interpolation='bilinear')
        plt.title(f'Topic {i} - TBBT')
        plt.axis('off')

        plt.tight_layout()
        plt.show()




if __name__ == '__main__':
    seasons_tvd = get_episode_scripts('corpus the vampire diaries')
    seasons_tbbt=get_episode_scripts('corpus the big bang theory')
    #THE VAMPIRE DIARIES
    # print_season_topics(seasons_tvd, 'lda')
    # print_season_topics(seasons_tvd, 'nmf')
    # print_global_topics_(seasons_tvd,'lda')
    # print_global_topics_(seasons_tvd,'nmf')
    #THE BIG BANG THEORY
    # print_season_topics(seasons_tbbt,'lda')
    # print_season_topics(seasons_tbbt,'nmf')
    # print_global_topics_(seasons_tbbt,'lda')
    # print_global_topics_(seasons_tbbt,'nmf')

    # glob_topics_tbbt_nmf = print_global_topics_(seasons_tbbt, 'nmf')
    # glob_topics_tvd_nmf = print_global_topics_(seasons_tvd, 'nmf')
    # create_wordcloud(glob_topics_tvd_nmf)
    # create_wordcloud(glob_topics_tbbt_nmf)
    # create_combined_wordclouds(glob_topics_tvd_nmf,glob_topics_tbbt_nmf)



