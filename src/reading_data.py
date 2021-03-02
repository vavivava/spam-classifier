import pandas as pd
from io import StringIO
from nltk.corpus import stopwords
import wordcloud
import matplotlib. pyplot as plt


class ReadingData:

    def reading_data(self, file_name, num_symbols):

        f = open(file_name, "r")
        data = f.read(num_symbols)
        data1 = StringIO(data)
        df = pd.read_csv(data1, sep="\t", header=None)
        df1 = df.rename(columns={0: 'target', 1: 'text'})
        df1['target'] = df1['target'].map({'spam': 1, 'ham': 0})

        return df1

    def convert_text(self, text_body):

        data1 = StringIO(text_body)
        df = pd.read_csv(data1, sep="\t", header=None)
        df2 = df.rename(columns={0: 'text'})

        return df2

    def plot_wordcloud(self, df):

        text = ' '.join(df['text'].astype(str).tolist())
        stopwords_data = set(wordcloud.STOPWORDS)
        fig_wordcloud = wordcloud.WordCloud(stopwords=stopwords_data, background_color="#ffa78c",
                                            width=3000, height=2000).generate(text)
        plt.figure(frameon=False)
        plt.imshow(fig_wordcloud)
        plt.axis('off')
        plt.savefig('wordcloud_spam.jpg')
