from joblib import load

class Model:

    def __init__(self):
        self.model = load("svcpipeline_v2.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result
