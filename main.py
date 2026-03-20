from src.cnn.train import train
from src.nlp.model import NLPModel

# CNN
model = train("data/processed")

# NLP
texts = ["irregular mole", "small round lesion"]
labels = [1, 0]

nlp = NLPModel()
nlp.train(texts, labels)

print("Pipeline initialized successfully")