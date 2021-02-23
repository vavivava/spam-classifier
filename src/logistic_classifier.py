from sklearn.linear_model import LogisticRegression


class LogisticClassifier:

    def logistic_classify(self, X_train, y_train, X_test, y_test, description, _C=1.0):

        model = LogisticRegression(C=_C).fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print('Test Score with', description, 'features', score)

        return model
