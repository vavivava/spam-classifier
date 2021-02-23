import pickle
from src.reading_data import ReadingData
from src.text_cleaned import TextCleaned
from src.prepare_models import PrepareModels


class MessagePredict:

    def message_predict(self, message, do_prediction):

        if do_prediction:
            pm = PrepareModels()
            pm.prepare_model()

        # Load model
        with open('text_classifier', 'rb') as training_model:
            model = pickle.load(training_model)

        with open('tfidf_model', 'rb') as training_model:
            tfidf_model = pickle.load(training_model)

        # Take input from user
        text_body = message

        rd = ReadingData()
        text_df = rd.convert_text(text_body)
        tc = TextCleaned()
        text_df['text_cleaned'] = list(map(tc.cleaning_text, text_df.text))
        text_df_cleaned = text_df['text_cleaned']
        text_vec = tfidf_model.transform(text_df_cleaned.values)

        # Predict custom message
        result = model.predict(text_vec)
        print(result)

        return result
