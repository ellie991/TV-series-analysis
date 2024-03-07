import os
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


excluded_words_tvd=['damon', 'elena', 'stefan', 'caroline', 'klaus', 'jeremy', 'bonnie', 'katherine', 'silas',
                   'elijah', 'enzo', 'lexi', 'tyler', 'kai', 'alaric', 'rebekah', 'anna',
                   'jenna', 'hayley', 'mason', 'sybil', 'vicki', 'jo', 'john', 'mickael', 'qetsiyah', 'lillian',
                   'liv', 'kol', 'mikaelson', 'isobel', 'nora', 'april', 'liz', 'luka', 'mary', 'emily', 'zach',
                   'julian', 'nadia', 'cade', 'logan', 'seline', 'connor', 'markos', 'bill', 'jules',
                   'andie', 'esther', 'lucy', 'ben', 'matt', 'aaron', 'finn', 'darren', 'james',
                   'well', 'right', 'okay', 'frederick', 'josie', 'helen', 'louise', 'noah', 'sarah', 'liam',
                   'lookwood', 'well', 'right', 'okay', 'look', 'say', 'tell', 'told', 'go', 'get', 'know', 'hi', 'oh',
                   'want', 'were', 'been', 'have', 'done', 'had', 'say', 'said', 'went', 'gone', 'can', 'could',
                   'been able', 'got', 'gotten', 'make', 'made', 'knew', 'known', 'will', 'would', 'think', 'thought', 'take',
                   'took', 'taken', 'see', 'saw', 'seen', 'came', 'come', 'want', 'wanted', 'looked', 'use',
                   'used', 'find', 'found', 'worked', 'called', 'tried', 'need', 'needed', 'give', 'gave'
                   'feel', 'felt', 'become', 'became', 'move', 'moved', 'thank', 'thanks']


excluded_words_tbbt = [ 'sheldon', 'leonard', 'penny', 'howard', 'raj', 'bernadette', 'amy', 'lesley',
                   'beverley', 'ramona', 'alicia', 'steph', 'kripke', 'yes', 'no', 'david', 'stephanie',
                   'gablehouser', 'cooper', 'like', 'hee', 'leslie', 'missy', 'need', 'let', 'take',
                   'would', 'could', 'come', 'bring', 'find', 'elizabeth', 'stan', 'zack', 'lee',
                   'hi', 'see', 'priya', 'mr', '...', 'make', 'hey', 'really', 'mean', 'one', 'jimmy',
                   'yeah', 'okay', 'stuart', 'well', 'look', 'right', 'told', 'tell', 'go', 'get',
                   'know', 'lucy', 'alex', 'oontz','arthur', 'janine', 'wolowitz', 'hello', 'wil',
                   'dan', 'lorvis', 'emily', 'hofstadter', 'kevin', 'come', 'josh', 'amelia', 'abbott',
                   'listen', 'roger', 'mary', 'claire', 'give', 'ask', 'alfred', 'play', 'call', 'bert',
                   'isabella', 'susan', 'stop', 'good', 'beverley', 'koothrappali', 'lalita', 'bye', 'talk',
                   'michaela', 'tom', 'wyatt', 'eat', 'glenn', 'two', 'try', 'open', 'christie', 'shelly',
                   'were', 'been', 'have', 'done', 'had', 'say', 'said', 'went', 'gone', 'can', 'could',
                   'been able', 'got', 'gotten', 'make', 'made', 'knew', 'known', 'will', 'would', 'think', 'thought', 'take',
                   'took', 'taken', 'see', 'saw', 'seen', 'came', 'come', 'want', 'wanted', 'looked', 'use',
                   'used', 'find', 'found', 'worked', 'called', 'tried', 'need', 'needed', 'give', 'gave'
                   'feel', 'felt', 'become', 'became', 'move', 'moved', 'sorri', 'thank']

excluded_words_words = ['stop', 'something', 'really', 'little', 'good', 'sorry',
             'sure', 'table', 'fine', 'first', 'melunaweh', 'beep', 'recent', 'phone',
             'mood', 'matter', 'thank', 'early','truly','otherwise','semi',
             'bathrobe', 'else', 'mail', 'brightly', 'bell', 'bathroom', 'entry', 'true',
             'item', 'back', 'away', 'around', 'still', 'behind', 'never', 'even', 'front', 'anything', 'thing',
             'last', 'everything', 'another', 'much','neither','whether','anyone','zero','throw',
             'maybe', 'next', 'nothing', 'someone', 'toward', 'suddenly', 'anybody', 'towards', 'since',
             'anyway', 'within', 'almost', 'wherever', 'throw', 'everywhere', 'highly', 'midly', 'sale',
             'eventually','except','ordinary', 'device','across', "ring", "whoa",
             'ungh','maybe','aside','every','woah','goodbye', 'shoe','afternoon','anytime', "blah", "great","rebecca",
             'others', 'whenever', "zero", "one", "two", "three", "four", "five", "knock",
             "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
             "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twentyone", "twentytwo", "twentythree", "twentyfour", "twentyfive",
             "twentysix", "twentyseven", "twentyeight", "twentynine", "thirty", "thirtyone", "thirtytwo", "thirtythree", "thirtyfour", "thirtyfive",
             "thirtysix", "thirtyseven", "thirtyeight", "thirtynine", "forty", "fortyone", "fortytwo", "fortythree", "fortyfour", "fortyfive",
             "fortysix", "fortyseven", "fortyeight", "fortynine", "fifty", "fiftyone", "fiftytwo", "fiftythree", "fiftyfour", "fiftyfive",
             "fiftysix", "fiftyseven", "fiftyeight", "fiftynine", "sixty", "sixtyone", "sixtytwo", "sixtythree", "sixtyfour", "sixtyfive",
             "sixtysix", "sixtyseven", "sixtyeight", "sixtynine", "seventy", "seventyone", "seventytwo", "seventythree", "seventyfour",
             "seventyfive", "seventysix", "seventyseven", "seventyeight", "seventynine", "eighty",
             "eightyone", "eightytwo", "eightythree", "eightyfour", "eightyfive", "eightysix", "eightyseven", "eightyeight", "eightynine",
             "ninety", "ninetyone", "ninetytwo", "ninetythree", "ninetyfour", "ninetyfive", "ninetysix", "ninetyseven", "ninetyeight",
             "ninetynine", "hundred", 'look', 'know', 'walk', 'like', 'take', 'come', 'need', 'going', 'go', 'make', 'turn', 'tell', 'would',
         'could', 'find', 'give','spell','speak'
         'looking', 'mean', 'walking', 'keep', 'talk', 'said', 'call', 'thought', 'trying', 'talking', 'sitting',
         'told','seems','stand','become','becomes',
         'answering','thinking', 'caused', 'putting', 'pulling', 'spoke', 'arrives', 'enters', 'seemed',
         'seen', 'pull', 'push', 'continues', 'washing','speaks','sits','turned','shall',
         'try', 'open', 'christie', 'shelly',
         'were', 'been', 'have', 'done', 'had', 'say', 'said', 'went', 'gone', 'can', 'could',
         'been able', 'got', 'gotten', 'make', 'made', 'knew', 'known', 'will', 'would', 'think', 'thought', 'take',
         'took', 'taken', 'see', 'saw', 'seen', 'came', 'come', 'want', 'wanted', 'looked', 'use',
         'used', 'find', 'found', 'worked', 'called', 'tried', 'need', 'needed', 'give', 'gave',
         'feel', 'felt', 'become', 'became', 'move', 'moved', 'sorri', 'thank']

excluded_words_tvd += excluded_words_words
excluded_words_tbbt += excluded_words_words

def preprocess_text(text, excluded_words):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(text)
    clean_tokens = []

    for word in tokens:
        clean_word = re.sub(r'[^a-zA-Z]', '', word.lower())
        if clean_word not in excluded_words:
            clean_word = lemmatizer.lemmatize(clean_word)

            if clean_word not in stop_words and 3 < len(clean_word) < 15:
                clean_tokens.append(clean_word)

    return clean_tokens


def perform_topic_modeling(corpus_path, visualization_path, excluded_words):
    for season in os.listdir(corpus_path):
        season_path = os.path.join(corpus_path, season)
        if os.path.isdir(season_path):
            documents = []
            for episode_file in os.listdir(season_path):
                episode_path = os.path.join(season_path, episode_file)
                if os.path.isfile(episode_path):
                    with open(episode_path, 'r', encoding='latin-1') as file:
                        text = file.read()
                        processed_text = preprocess_text(text, excluded_words)
                        documents.append(processed_text)

            lda_model_path = os.path.join(season_path, 'lda_model')

            if not os.path.exists(lda_model_path):
                os.makedirs(lda_model_path)

            dictionary = corpora.Dictionary(documents)
            corpus = [dictionary.doc2bow(doc) for doc in documents]

            lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=20)

            lda_model_file_path = os.path.join(lda_model_path, 'model.lda')
            lda_model.save(lda_model_file_path)

            vis_data = gensimvis.prepare(lda_model, corpus, dictionary)

            output_folder = os.path.join(visualization_path, f"Topic Modeling {season.lower()}")
            os.makedirs(output_folder, exist_ok=True)

            lda_visualization_folder = os.path.join(output_folder, 'lda_visualization')
            os.makedirs(lda_visualization_folder, exist_ok=True)

            output_file = os.path.join(lda_visualization_folder, 'index.html')
            pyLDAvis.save_html(vis_data, output_file)

            print(f'Visualizzazione salvata per la stagione {season} in {output_file}')




def perform_topic_modeling_global(documents, visualization_path):
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=20)
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)

    output_file = os.path.join(visualization_path, 'global_lda_visualization.html')
    pyLDAvis.save_html(vis_data, output_file)
    print(f'Global LDA visualization saved at: {output_file}')
def collect_documents(corpus_path, excluded_words):
    documents = []
    for season in os.listdir(corpus_path):
        season_path = os.path.join(corpus_path, season)
        if os.path.isdir(season_path):
            for episode_file in os.listdir(season_path):
                episode_path = os.path.join(season_path, episode_file)
                if os.path.isfile(episode_path):
                    with open(episode_path, 'r', encoding='latin-1') as file:
                        text = file.read()
                        processed_text = preprocess_text(text, excluded_words)
                        documents.append(processed_text)
    return documents




if __name__=='__main__':
    # The Big Bang Theory
    corpus_path_tbbt = 'corpus the big bang theory'
    visualization_path_tbbt = 'topic_modeling_tbbt'
    perform_topic_modeling(corpus_path_tbbt, visualization_path_tbbt, excluded_words_tbbt)
    global_documents_tbbt = collect_documents('corpus the big bang theory', excluded_words_tbbt)
    perform_topic_modeling_global(global_documents_tbbt, visualization_path_tbbt)



    # The Vampire Diaries
    corpus_path_tvd = 'corpus the vampire diaries'
    visualization_path_tvd = 'topic_modeling_tvd'
    perform_topic_modeling(corpus_path_tvd, visualization_path_tvd, excluded_words_tvd)
    global_documents_tvd = collect_documents('corpus the vampire diaries', excluded_words_tvd)
    perform_topic_modeling_global(global_documents_tvd, visualization_path_tvd)
