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

- Web Scraping: BeautifulSoup, Scrapy
- Sentiment Analysis: TextBlob, VADER
- Topic Modeling: gensim, scikit-learn
- Data Processing and Visualization: pandas, numpy, matplotlib, seaborn

Repository Structure

- `data/`: Contains the raw and processed data used in the analysis.
- `code/`: Includes Python scripts for each step of the analysis.
- `reports/`: Contains the final summary report in PDF format.

How to Run the Code

To run the scripts, ensure that all dependencies listed in `requirements.txt` are installed. You can execute each script individually, following the order suggested in the project overview.

Results

Our analysis revealed key differences in the sentiment trends and topics between the two series, reflecting their distinct narrative styles and character development arcs. The final report summarizing these results is available in the `reports/` folder. (Please note: The report is in Italian.)
