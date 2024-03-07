from matplotlib.patches import Patch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re



#Funzione per il calcolo della correlazione tra i valori numerici ottenuti
# con il metodo vader e quelli ottenuti con il metodo textblob, per verificare se
#i due metodi convergono o meno.

def calculate_correlation(df):
    correlation=df['Vader_sent_numeric'].corr(df['Textblob_sent_numeric'])
    print(f"Correlation between Vader values and TextBlob values: {round(correlation,5)}")
    return

def extract_episode_number(episode):

    match = re.search(r'\d+', episode)
    if match:
        return int(match.group())
    return 0  # Re

#Funzione che va a visualizzare graficamente utilizzando i valori numerici di vader
# l'andamento di ciascun personaggio stagione per stagione.
#Crea un grafico con 3 linee, ognuna delle quali rappresenta il sentiment
#dei personaggi.

def sentiment_line_for_episodes_vader(df):
    season_character_sentiment = df.groupby(['Season', 'Character', 'Episode'])['Vader_sent_numeric'].mean().reset_index()
    seasons = season_character_sentiment['Season'].unique()
    for season in seasons:
        plt.figure(figsize=(10, 6))

        season_data = season_character_sentiment[season_character_sentiment['Season'] == season]

        for character in season_data['Character'].unique():
            character_data = season_data[season_data['Character'] == character]
            character_data_copy = character_data.copy()
            character_data_copy["Episode_number"] = character_data_copy['Episode'].apply(extract_episode_number)

            character_data=character_data_copy.sort_values(by="Episode_number")
            plt.plot(character_data_copy['Episode'], character_data_copy['Vader_sent_numeric'], label=character)

        plt.xlabel('Episode')
        plt.ylabel('Sentiment Numeric Value (Vader)')
        plt.title(f'Sentiment Trend for Season {season} (Vader)')
        plt.axhline(y=0, color='green', linestyle='--', label='Neutral')
        plt.legend()
        plt.ylim(-0.6,0.6)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=90)
        plt.show()

#Funzione che va a visualizzare graficamente utilizzando i valori numerici di vader
# l'andamento di ciascun personaggio stagione per stagione.
#Crea un grafico con 3 linee, ognuna delle quali rappresenta il sentiment
#dei personaggi.
def sentiment_line_for_episodes_textb(df):
    season_character_sentiment = df.groupby(['Season', 'Character', 'Episode'])['Textblob_sent_numeric'].mean().reset_index()
    seasons = season_character_sentiment['Season'].unique()

    for season in seasons:
        plt.figure(figsize=(10, 6))
        season_data = season_character_sentiment[season_character_sentiment['Season'] == season]

        for character in season_data['Character'].unique():
            character_data = season_data[season_data['Character'] == character]
            character_data_copy = character_data.copy()
            character_data_copy["Episode_number"] = character_data_copy['Episode'].apply(extract_episode_number)

            character_data=character_data_copy.sort_values(by="Episode_number")
            plt.plot(character_data_copy['Episode'], character_data_copy['Textblob_sent_numeric'], label=character)

        plt.xlabel('Episode')
        plt.ylabel('Sentiment Numeric Value (TextBlob)')
        plt.title(f'Sentiment Trend for Season {season} (TextBlob)')
        plt.axhline(y=0, color='green', linestyle='--', label='Neutral')
        plt.legend()
        plt.ylim(-0.6,0.6)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=90)
        plt.show()

#Funzione per creare un diagramma a barre per ogni
#stagione che permette di conteggiare quanti episodi positivi e quanti negativi ci sono per ogni personaggio.

def vader_barplot_by_season(df):
    counts = df.groupby(['Season', 'Character', 'Vader_sent_label']).size().unstack(fill_value=0)

    seasons = df['Season'].unique()
    for season in seasons:
        plt.figure(figsize=(10, 6))
        season_data = counts.loc[season]
        characters = season_data.index.unique()
        for character in characters:
            character_data = season_data.loc[character]
            plt.bar(character, character_data['positive'], label=f'{character} - Positive', color='lightgreen')
            plt.bar(character, character_data['negative'], bottom=character_data['positive'], label=f'{character} - Negative', color='lightcoral')


        plt.xlabel('Character')
        plt.ylabel('Count of Episodes')
        plt.title(f'Vader Sentiment by Character for {season}')
        plt.legend(handles=[
            Patch(color='lightgreen', label='Positive'),
            Patch(color='lightcoral', label='Negative')
        ], loc='upper right')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

#Grafico che crea dei boxplot per vedere la variabilita del sentiment
#durante le stagioni, per capire quale è il personaggio con maggiore
#variabilita
def box_plot_episodes(df):
    custom_palette = sns.color_palette("Set2")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Character', y='Vader_sent_numeric',palette=custom_palette)
    plt.xlabel('Character')
    plt.ylabel('Vader Sentiment Numeric Value')
    plt.title('Variability of Vader Sentiment by Character')
    plt.xticks(rotation=0)
    plt.axhline(y=0, color='green', linestyle='--', label='Neutral')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


#ora utilzziamo un grafico a barre per rappresentare il sentiment
#di ogni personaggio cumulato per tutte le stagioni.


def plot_combined_sentiment(df):
    categorie_sentiment = ['positive', 'neutral', 'negative']
    colori_sentiment = {'positive': 'lightgreen', 'neutral': 'lightsalmon', 'negative': 'lightcoral'}
    personaggi_unici = df['Character'].unique()

    sentiment_cumulativo = {categoria: [0] * len(personaggi_unici) for categoria in categorie_sentiment}

    for i, personaggio in enumerate(personaggi_unici):

        df_personaggio = df[df['Character'] == personaggio]

        for categoria in categorie_sentiment:
            conteggio = len(df_personaggio[df_personaggio['Vader_sent_label'] == categoria])
            sentiment_cumulativo[categoria][i] = conteggio
    x = np.arange(len(personaggi_unici))
    larghezza_barre = 0.2

    for i, categoria in enumerate(categorie_sentiment):
        altezza_barre = sentiment_cumulativo[categoria]
        colori = [colori_sentiment[categoria]] * len(personaggi_unici)
        plt.bar(x + i * larghezza_barre, altezza_barre, larghezza_barre, label=categoria, color=colori)

    plt.xticks(x + larghezza_barre * ((len(categorie_sentiment) - 1) / 2), personaggi_unici, rotation=45)
    plt.xlabel('Character')
    plt.ylabel('Cumulative')
    plt.title('Sentiment for Character')
    plt.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


#Grafico per analizzare il trend di ciascun personaggio separatamente,
#visualizzando per ogni stagione il numero di episodi positivi e negativi


def barplot_for_character(df):
    characters = df['Character'].unique()

    for character in characters:
        plt.figure(figsize=(10, 6))
        df_character = df[df['Character'] == character]
        counts = df_character.groupby(['Season', 'Vader_sent_label']).size().unstack(fill_value=0)

        seasons = counts.index
        positive_count = counts['positive']
        negative_count = counts['negative']

        bar_width = 0.4
        x = range(len(seasons))

        plt.bar(x, positive_count, width=bar_width, label='Positive', color='lightgreen')

        plt.bar([pos + bar_width for pos in x], negative_count, width=bar_width, label='Negative',
                color='lightcoral')

        plt.xlabel('Seasons')
        plt.ylabel('Episode count')
        plt.title(f'Episode Count (Positive and Negative) for {character}')
        plt.xticks([pos + bar_width / 2 for pos in x], seasons)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()







#Funzione che crea  un grafico a linee in cui ogni linea rappresenta il trend
#del sentiment tra le stagioni per ogni personaggio
def plot_line_for_seasons(df):
    characters = df['Character'].unique()

    plt.figure(figsize=(12, 6))

    for character in characters:
        df_character = df[df['Character'] == character]
        plt.plot(df_character['Season'], df_character['Vader_sent_numeric'], label=character)

    plt.xlabel('Season')
    plt.ylabel('Sentiment Numeric Value (Vader)')
    plt.title('Sentiment Trend by Season (Vader) for Characters')
    plt.legend()
    plt.ylim(-0.15,0.15)
    plt.axhline(y=0, color='green', linestyle='--', label='Neutral')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.show()

#Funzione che crea un grafico a barre  per il conteggio delle stagioni positive e negative

def plot_combined_sentiment_seasons(df):

    sentiment_categories = ['positive', 'neutral', 'negative']
    sentiment_colors = {'positive': 'lightgreen', 'neutral': 'lightsalmon', 'negative': 'lightcoral'}

    chracters = df['Character'].unique()
    cumulative_sentiment = {category: [0] * len(chracters) for category in sentiment_categories}
    for i, chracter in enumerate(chracters):
        df_character = df[df['Character'] == chracter]
        for category in sentiment_categories:
            count = len(df_character[df_character['Vader_sent_label'] == category])
            cumulative_sentiment[category][i] = count
    x = np.arange(len(chracters))
    bar_width = 0.2

    for i, category in enumerate(sentiment_categories):
        bar_lenght = cumulative_sentiment[category]
        colors = [sentiment_colors[category]] * len(chracters)
        plt.bar(x + i * bar_width, bar_lenght, bar_width, label=category, color=colors)

    plt.xticks(x + bar_width * ((len(sentiment_categories) - 1) / 2), chracters, rotation=45)
    plt.xlabel('Character')
    plt.ylabel('Cumulative')
    plt.title('Sentiment for Character')
    plt.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()



if __name__=='__main__':
    df_tvd=pd.read_csv('df/3_episode_results_tbbt.csv')
    df_tbbt=pd.read_csv('df/3_episode_results_tbbt.csv')
    # calculate_correlation(df_tvd)
    # calculate_correlation(df_tbbt)
    #1
    # sentiment_line_for_episodes_vader(df_tvd)
    # sentiment_line_for_episodes_textb(df_tvd)
    # sentiment_line_for_episodes_vader(df_tbbt)
    # sentiment_line_for_episodes_textb(df_tbbt)

    #2
    # vader_barplot_by_season(df_tvd)
    # vader_barplot_by_season(df_tbbt)
    #3
    # box_plot_episodes(df_tvd)
    # box_plot_episodes(df_tbbt)
    #4
    # barplot_for_character(df_tvd)
    # barplot_for_character(df_tvd)
    #5
    # plot_combined_sentiment(df_tvd)
    # plot_combined_sentiment(df_tbbt)

    df_tvd=pd.read_csv('df/3_season_results_tvd.csv')
    df_tbbt=pd.read_csv('df/3_season_results_tbbt.csv')
    calculate_correlation(df_tvd)
    calculate_correlation(df_tbbt)
    plot_line_for_seasons(df_tvd)
    plot_combined_sentiment_seasons(df_tvd)
