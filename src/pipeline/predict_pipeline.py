
import pickle

class PredictPipeline:
    def __init__(self):
        self.model = pickle.load(open("artifacts/model.pkl", "rb"))
        self.pre = pickle.load(open("artifacts/preprocessor.pkl", "rb"))

    def predict(self, data):
        data = self.pre.transform(data)
        return self.model.predict(data)
