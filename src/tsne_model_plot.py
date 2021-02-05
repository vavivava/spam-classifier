from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

# -----------------Create TSNE model and plot it -----------

class TsneModelPlot:

    def __init__(self):
        pass

    def tsne_plot(self, model):

        labels = []
        tokens = []

        for word in model.wv.vocab:
            tokens.append(model[word])
            labels.append(word)

        tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(tokens)

        x_data = []
        y_data = []
        for value in new_values:
            x_data.append(value[0])
            y_data.append(value[1])

        plt.figure(figsize=(18, 18))

        for i in range(len(x_data)):
            plt.scatter(x_data[i], y_data[i])
            plt.annotate(labels[i],
                         xy=(x_data[i], y_data[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

        plt.show()
