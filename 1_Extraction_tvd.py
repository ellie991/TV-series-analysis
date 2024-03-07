import os
import requests
from bs4 import BeautifulSoup


#Funzione che prende in ingresso la pagina principale contenente gli script
#della serie e restiuisce un dizionario contenente come
#chiavi le stagioni e come valori le liste contenenti i link delle punatate
#di ciascuna stagione.
def link_extractor(url):
    try:
        base_url='https://vampirediaries.fandom.com'
        page=requests.get(url)
        if page.status_code!=200:
            raise Exception(f'Error: {page.status_code}')
        parser=BeautifulSoup(page.content ,'html.parser')
        seasons=parser.find_all('div', style='padding:0em 0.25em')
        season_links={}
        for season in seasons:
            episodes_link=[]
            episodes=season.find_all('a',href=True)
            for episode in episodes:
                if 'Season' in episode['title'] and episode['title'] not in season_links and 'Originals' not in episode['title']:
                    season_links[episode['title']]=episodes_link
                elif 'Transcript' in episode['title']:
                    episodes_link.append(f"{base_url}{episode['href']}")
        return season_links
    except Exception as e:
        print(f'Page not found {str(e)}')


#Funzione che prende in ingresso il link di un episodio, la directory
#che rappresenta la stagione di appartenenza e il numero dell'episodio
#e estrae il testo utile dalla pagina salvandolo in un file.txt

def extracting_texts(episode,season_dir,episode_number):
    try:
        page = requests.get(episode)
        parser = BeautifulSoup(page.text, 'html.parser')
        divs=parser.find_all('div',class_="mw-parser-output")

        for i, div in enumerate(divs):
            with open(os.path.join(season_dir,f'episode_{episode_number}.txt'), 'w', encoding='utf-8') as file:
                paragraphs = div.find_all(['p', 'dd'])
                script_text = ''
                for paragraph in paragraphs:
                    if paragraph.name == 'p' and '[' in paragraph.get_text() and not paragraph.get_text().isupper() or '(' in paragraph.get_text():
                        script_text += paragraph.get_text().replace('(', '[').replace(')', ']')+'\n'
                    elif paragraph.name == 'dd':
                        script_text += paragraph.get_text().replace('(', '[').replace(')', ']') + '\n'

                script_text = '\n'.join(line.replace('\u00A0', ' ') for line in script_text.splitlines() if line.strip())
                file.write(script_text)
    except Exception as e:
        print(f'Error for episode {episode_number}: {str(e)}')


#Funzione che crea il corpus di testi: prende in ingresso il risultato di link_extractor(url),
#crea la directory principale del corpus, successivamente crea una directory per ogni stagione,
# e inserisce all'interno delle stesse, iterando tra gli episodi, i file.txt creati dalla funzione
#extracting_texts

def create_corpus(d):
    if not os.path.exists('corpus the vampire diaries'):
        os.mkdir('corpus the vampire diaries')
    for season_number,season_links, in enumerate(d, start=1):
        season_dir = os.path.join('corpus the vampire diaries', f'Season{str(season_number).zfill(2)}')
        if not os.path.exists(season_dir):
            os.mkdir(season_dir)
        for episode_number,episode_url in enumerate(d[season_links],start=1):

            extracting_texts(episode_url,season_dir,episode_number)

    return


if __name__=='__main__':
    url='https://vampirediaries.fandom.com/wiki/Category:Episode_Transcripts'
    d=link_extractor(url)
    create_corpus(d)
