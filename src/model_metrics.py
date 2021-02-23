import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve


class ModelMetrics:

    def __init__(self):
        pass

    def model_metrics(self, y_test, y_predicted):

        tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()
        print(pd.DataFrame(confusion_matrix(y_test, y_predicted), columns=['Predicted Spam', "Predicted Ham"],
                           index=['Actual Spam', 'Actual Ham']))

        print(f'\nTrue Positives: {tp}')
        print(f'False Positives: {fp}')
        print(f'True Negatives: {tn}')
        print(f'False Negatives: {fn}')
        print(f'True Positive Rate: { (tp / (tp + fn))}')
        print(f'Specificity: { (tn / (tn + fp))}')
        print(f'False Positive Rate: { (fp / (fp + tn))}')
        print(f"    ")
        print(roc_auc_score(y_test, y_predicted))
        print(classification_report(y_test, y_predicted))
