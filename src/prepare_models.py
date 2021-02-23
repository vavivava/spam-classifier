from src.text_cleaned import TextCleaned
from src.model_metrics import ModelMetrics
from src.reading_data import ReadingData
from src.logistic_classifier import LogisticClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import pickle
import os


class PrepareModels:

    def prepare_model(self):

        # Reading dataset
        script_dir = os.path.dirname(__file__)
        rel_path = 'SMSSpamCollection.txt'
        abs_file_path = os.path.join(script_dir, rel_path)

        file_name = abs_file_path
        num_symbols = 20000000
        rd = ReadingData()
        df = rd.reading_data(file_name, num_symbols)
        data_spam = df[df['target'] == 1].copy()
        rd.plot_wordcloud(data_spam)

        # Cleaning dataset
        tc = TextCleaned()
        df['text_cleaned'] = list(map(tc.cleaning_text, df.text))
        text_cleaned = df['text_cleaned']

        # Convert values to TFIDF values
        tfidf_vectorized = TfidfVectorizer(use_idf=True, max_df=0.95)
        X_data = tfidf_vectorized.fit_transform(text_cleaned.values)

        # Prepare logistic model
        X_train, X_test, y_train, y_test = train_test_split(X_data, df["target"], train_size=0.7, random_state=0)

        lc = LogisticClassifier()
        model = lc.logistic_classify(X_train, y_train, X_test, y_test, 'tf-idf')

        model.fit(X_train, y_train)
        y_predicted_tfidf = model.predict(X_test)

        mm = ModelMetrics()
        metrics = mm.model_metrics(y_test, y_predicted_tfidf)
        print(metrics)

        # Save models
        with open('text_classifier', 'wb') as picklefile:
            pickle.dump(model, picklefile)

        with open('tfidf_model', 'wb') as picklefile:
            pickle.dump(tfidf_vectorized, picklefile)

        print("Models were updated successfully")
