from sklearn.preprocessing import StandardScaler

class OneClassSum(object):
    def fit(self, X, y=None, sample_weight=None):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        S = X.sum(axis=1)
        return S