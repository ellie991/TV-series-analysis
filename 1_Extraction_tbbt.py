import os
import requests
from bs4 import BeautifulSoup


#Funzione che prende in ingresso la pagina principale contenente gli script
#della serie e restiuisce un dizionario contenente come
#chiavi le stagioni e come valori le liste contenenti i link delle punatate
#di ciascuna stagione.
def link_extractor(url):
    base_url = 'https://bigbangtrans.wordpress.com'
    page = requests.get(url)
    parser = BeautifulSoup(page.content, 'html.parser')
    episode_links = parser.find_all('a', href=True)

    season_links = {}
    links=[]
    for episode in episode_links:
        if 'Series' in episode.get_text():
            season=episode.get_text().split()
            if f'Season{season[1]}' not in season_links:
                links=[]
                season_links[f'Season{season[1]}']=links
                links.append(episode['href'])
            else:
                links.append(episode['href'])
    return season_links



#Funzione che prende in ingresso il link di un episodio, la directory
#che rappresenta la stagione di appartenenza e il numero dell'episodio
#e estrae il testo utile dalla pagina salvandolo in un file.txt
def extracting_texts(episode,season_dir,episode_number):
    page = requests.get(episode)
    parser = BeautifulSoup(page.text, 'html.parser')
    divs = parser.find_all('div', class_="entrytext")
    with open(os.path.join(season_dir, f'episode_{episode_number}.txt'), 'w', encoding='utf-8') as file:
        for div in divs:
            paragraphs = div.find_all(['p'])
            script_text=''
            for paragraph in paragraphs:
                if ':' in paragraph.get_text() and 'TwitterFacebook' not in paragraph.get_text():
                    script_text+=paragraph.get_text()+'\n'
                elif '(' in paragraph.get_text():
                    script_text += paragraph.get_text()+'\n'

            cleaned_text = ''
            for row in script_text:
                cleaned_text = script_text.replace('\u00A0', ' ').replace('’', ' ').replace('…', ' ')
            file.write(cleaned_text)

    return cleaned_text

#Funzione che crea il corpus di testi: prende in ingresso il risultato di link_extractor(url),
#crea la directory principale del corpus, successivamente crea una directory per ogni stagione,
# e inserisce all'interno delle stesse, iterando tra gli episodi, i file.txt creati dalla funzione
#extracting_texts


def create_corpus(d):
    if not os.path.exists('corpus the big bang theory'):
        os.mkdir('corpus the big bang theory')
    for season in d:
        season_dir = os.path.join('corpus the big bang theory', season)
        if not os.path.exists(season_dir):
            os.mkdir(season_dir)
        for episode_number,episode_url in enumerate(d[season],start=1):
            extracting_texts(episode_url,season_dir,episode_number)



if __name__=='__main__':
    url='https://bigbangtrans.wordpress.com'
    d=link_extractor(url)
    create_corpus(d)
