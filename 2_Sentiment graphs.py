import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



#Funzione per il calcolo della correlazione tra i valori numerici ottenuti
# con il metodo vader e quelli ottenuti con il metodo textblob, per verificare se
#i due metodi convergono o meno.

def calculate_correlation(df):
    correlation=df['Vader_sent_numeric'].corr(df['Textblob_sent_numeric'])
    print(f"Correlation between Vader values and TextBlob values: {round(correlation,5)}")
    return



#1 Grafico a linee per visualizzare l'andamento del sentiment nel corso degli episodi
# durante ogni stagione, si utilizzano i valori numerici di vader e text blob
#una linea per vader, una per textblob.

def plot_episodes_line(df):
    seasons = df['Season'].unique()
    for season in seasons:
        season_data = df[df['Season'] == season]

        plt.figure(figsize=(10, 6))
        plt.plot(season_data['Episode'], season_data['Vader_sent_numeric'], label='Vader sent')
        plt.plot(season_data['Episode'], season_data['Textblob_sent_numeric'], label='TextBlob sent')
        plt.xlabel('Episode')
        plt.xticks(rotation=90)
        plt.axhline(y=0, color='green', linestyle='--', label='Neutral')
        plt.ylim(-0.15, 0.15)
        plt.ylabel('Sentiment')
        plt.title(f'Trend sentiment {season}')
        plt.legend()
        plt.show()







#2 Grafico a barre per visualizzare per ogni stagione il contaggio degli episodi
# positivi e negativi.

def plot_sentiment_for_episode_bar(df):
    unique_seasons = df['Season'].unique()
    sentiment_categories = ['positive', 'negative']
    season_counts = []
    for season in unique_seasons:
        season_data_vader = df[(df['Season'] == season)]['Vader_sent_label'].value_counts()

        for sentiment in sentiment_categories:
            if sentiment not in season_data_vader.index:
                season_data_vader[sentiment] = 0

        season_counts.append(
            {'Season': season, 'Positive': season_data_vader['positive'], 'Negative': season_data_vader['negative']})

    season_counts_df = pd.DataFrame(season_counts)
    width = 0.4
    x = np.arange(len(unique_seasons))
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, season_counts_df['Positive'], width, label='Positive', color='lightgreen')
    plt.bar(x + width / 2, season_counts_df['Negative'], width, label='Negative', color='lightcoral')
    plt.xlabel('Season')
    plt.xticks(x, unique_seasons, rotation=0)
    plt.ylabel('Number of episodes')
    plt.title('Number of positive/negative episodes for each season')

    plt.legend(title='Sentiment', loc='upper right')

    plt.show()

#3 Grafico che genera dei box plot per ogni stagione per mostrare
# la variabilita degli episodi, utilizzando i valori di vader tra le stagioni.

def plot_vader_variation_by_season(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Season', y='Vader_sent_numeric', palette='Set3')
    plt.axhline(y=0, color='green', linestyle='--', label='Neutral')
    plt.xlabel('Season')
    plt.xticks(rotation=90)
    plt.ylabel('Vader Sentiment Score')
    plt.title('Variability in Vader Sentiment Scores for Each Season')
    plt.show()


#4 Grafico a linee che permette di mostrare il trend del sentiment nel corso delle stagioni

def plot_sentiment_for_season_line(df):
    sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
    df['Vader_sentiment'] = df['Vader_sent_numeric'].map(sentiment_mapping)
    df['Textblob_sentiment'] = df['Textblob_sent_numeric'].map(sentiment_mapping)
    seasons = df['Season'].unique()
    plt.figure(figsize=(12, 6))
    vader_sentiment_values = df['Vader_sent_numeric'].tolist()
    textblob_sentiment_values = df['Textblob_sent_numeric'].tolist()
    plt.plot(seasons, vader_sentiment_values, label='Vader Sentiment', marker='o')
    plt.plot(seasons, textblob_sentiment_values, label='TextBlob Sentiment', marker='o')
    plt.title("Sentiment for Seasons")
    plt.xlabel('Season')
    plt.ylabel('Sentiment')
    plt.axhline(0, color='green', linestyle='--', linewidth=1.0)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

#Funzione che costruisce un grafico a barre che permette di visualizzare
#il conteggio del numero di stagioni con sentiment positivo e negativo.
def plot_season_bar(df):
    # Conteggio totale di stagioni positive e negative
    positive_count = len(df[df['Vader_sent_label'] == 'positive'])
    negative_count = len(df[df['Vader_sent_label'] == 'negative'])

    plt.figure(figsize=(8, 6))
    plt.bar(['Positive', 'Negative'], [positive_count, negative_count], color=['lightgreen', 'lightcoral'])
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Seasons')
    plt.title('Number of Positive and Negative Seasons')
    plt.show()





#Funzione che crea un grafico a linee che permette di visualizzare i trend del sentiment
#calcolati con vader per tvd e tbbt.
def seasons_tvd_vs_tbbt_vader(df_tvd,df_tbbt):
    plt.figure(figsize=(12, 6))
    plt.plot(df_tvd['Season'], df_tvd['Vader_sent_numeric'], label='The Vampire Diaries', marker='o')
    plt.plot(df_tbbt['Season'], df_tbbt['Vader_sent_numeric'], label='The Big Bang Theory', marker='o')

    plt.xlabel('Season')
    plt.ylabel('Vader Sentiment Score')
    plt.title('Comparison of Vader Sentiment Scores')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()




#Funzione che crea un grafico a linee che permette di visualizzare i trend del sentiment
#calcolati con textblob per tvd e tbbt.

def seasons_tvd_vs_tbbt_textb(df_tvd, df_tbbt):
    plt.figure(figsize=(12, 6))
    plt.plot(df_tvd['Season'], df_tvd['Textblob_sent_numeric'], label='The Vampire Diaries', marker='o')

    plt.plot(df_tbbt['Season'], df_tbbt['Textblob_sent_numeric'], label='The Big Bang Theory', marker='o')

    plt.xlabel('Season')
    plt.ylabel('TextBlob Sentiment Score')
    plt.title('Comparison of TextBlob Sentiment Scores')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()





if __name__=='__main__':
    #df_tbbt = pd.read_csv('df/2_episode_results_tbbt.csv')
    #df_tvd = pd.read_csv('df/2_episode_results_tvd.csv')
    #calculate_correlation(df_tbbt)
   # 1#
    #plot_episodes_line(df_tvd)
    #plot_episodes_line(df_tbbt)
    #2
    #plot_sentiment_for_episode_bar(df_tvd)
    #plot_sentiment_for_episode_bar(df_tbbt)
    #3#
    #plot_vader_variation_by_season(df_tvd)
    #plot_vader_variation_by_season(df_tbbt)
    df_tvd=pd.read_csv('df/2_season_results_tvd.csv')
    df_tbbt = pd.read_csv('df/2_season_results_tbbt.csv')

    #calculate_correlation(df_tvd)
    #4
    #plot_sentiment_for_season_line(df_tvd)
    #plot_sentiment_for_season_line(df_tbbt)
    #5
    #plot_season_bar(df_tvd)
    #plot_season_bar(df_tbbt)
    #6
    #seasons_tvd_vs_tbbt_vader(df_tvd,df_tbbt)
    #seasons_tvd_vs_tbbt_textb(df_tvd,df_tbbt)


























