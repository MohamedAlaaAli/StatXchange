import numpy as np
# First, let's create our Naive Bayes Classifier from scratch
class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)  # 0 or 1
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        for i, c in enumerate(self._classes):
            X_c = X[y == c]
            # Mean of each feature in each class
            self._mean[i, :] = X_c.mean(axis=0)
            self._var[i, :] = X_c.var(axis=0)
            self._priors[i] = X_c.shape[0] / \
                float(n_samples)  # Probability of each class

    def predict(self, X):
        Z = np.array(X)
        y_pred = [self._predict(x) for x in Z]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []
        for i, c in enumerate(self._classes):  # Calculate posterior for each class
            prior = np.log(self._priors[i])
            posterior = np.sum(np.log(self._pdf(i, x)))  # Gaussian model
            posterior += prior
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, i, x):
        mean = self._mean[i]
        var = self._var[i]

        numerator = np.exp(-(((x - mean) ** 2) / (2 * var)))
        doneminator = np.sqrt(2 * np.pi * var)
        return numerator / doneminator


# Let's create a function to calculate the accuracy of our NB Classifier
def accuracy(a, b):
    return np.sum(a == b) / len(a)


