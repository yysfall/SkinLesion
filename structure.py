import os

folders = [
    "data", "data/raw", "data/processed", "data/splits",
    "notebooks",
    "src", "src/cnn", "src/nlp", "src/rl",
    "experiments",
    "outputs", "outputs/models", "outputs/plots",
    "docs"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    open(os.path.join(folder, ".gitkeep"), "w").close()
