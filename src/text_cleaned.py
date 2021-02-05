import pandas as pd
import re
#from collections import defaultdict
import logging
import wordcloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import json

with open('patterns_replacements.json') as config_file:
    json_data = json.load(config_file)

class TextCleaned:

    def __init__(self):
        pass

    def cleaning_text(self, text, remove_stop_words=True):

        text = text.lower()

        for p in json_data['pairs']:
            text = re.sub(p["pattern"], p["replacement"], text, flags=re.IGNORECASE)

        text = re.sub(r'<br />', ' ', text)
        text = re.sub(r'\'', ' ', text)
        text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
        text = re.sub(r'&amp;', '', text)
        text = re.sub(r'\<a href', ' ', text)
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

        if remove_stop_words:
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)

        text = nltk.WordPunctTokenizer().tokenize(text)

        return text

# check the code
'''
mm = TextCleaned()
#print("Today I've been in the mall and bought some ice cream. I've a headache right now.")
text2 = mm.cleangingtext("Today I've been in the mall and bought some ice cream. I've a headache right now.")
print(text2)
'''
# ['today', 'mall', 'bought', 'ice', 'cream', 'headache', 'right']
