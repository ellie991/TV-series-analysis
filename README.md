This project, which analyzes two distinct TV series, The Big Bang Theory and The Vampire Diaries, was carried out by Elena Valdes and Manuela Scalas, students of the Master's degree program in Data Science, Business Analytics, and Innovation at the University of Cagliari. The project focuses on five main points:

1. Extraction of the transcripts of each series' episodes:
   We performed web scraping to collect the transcripts of all episodes for both TV series.

2. Sentiment Analysis of each series:
   We identified the sentiment of each episode and then of each season to detect the sentiment trend throughout the series' evolution.

3. Sentiment Analysis of three main characters for each series:
   We analyzed the dialogues of Stefan Salvatore, Damon Salvatore, and Elena Gilbert in The Vampire Diaries; and Sheldon Cooper, Leonard Hofstadter, and Penny in The Big Bang Theory. The goal was to understand the sentiment evolution of each character throughout the series.

4. Topic modeling:
   We used topic modeling techniques to identify the main topics discussed season by season and then compared the topics across the two series.

5. Comparison between the topics of the two series:
   We compared the topics identified in both series to understand similarities and differences in their narrative content.

Technologies and Libraries Used

- Web Scraping: BeautifulSoup
- Sentiment Analysis: TextBlob, VADER, nltk
- Topic Modeling: gensim, scikit-learn, pyLDAvis
- Data Processing and Visualization: pandas, numpy, matplotlib, seaborn

How to Run the Code

To run the scripts, ensure that all dependencies listed in `requirements.txt` are installed. You can execute each script individually, following the order suggested in the project overview.

Results

Our analysis revealed key differences in the sentiment trends and topics between the two series, reflecting their distinct narrative styles and character development arcs. SENTIMENTAL ANALYSIS (series and characters): The Vampire Diaries* exhibits a darker, more intense sentiment, reflecting its dramatic and supernatural themes; The Big Bang Theory shows a lighter, more positive sentiment, consistent with its comedic and everyday themes.
TOPIC MODELING: The Vampire Diaries focuses on themes such as romance, vengeance, and supernatural elements; The Big Bang Theory centers on science, relationships, and pop culture.
These findings highlight the distinct narrative and thematic approaches of the two series.
