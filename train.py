import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import random
import os

# Step 1: Load the blank model for POS tagging
nlp = spacy.blank("fa")  # Use 'en' or a language model if necessary

# Step 2: Load the training and testing data from .spacy files
train_data_path = "train"  # Directory with the train data in .spacy format
test_data_path = "test"    # Directory with the test data in .spacy format


def load_data(path):
    """Load .spacy files from a directory and extract POS tags"""
    data = []
    for filename in os.listdir(path):
        if filename.endswith(".spacy"):
            # Load the .spacy file
            doc_bin = spacy.tokens.DocBin().from_disk(os.path.join(path, filename))
            for doc in doc_bin.get_docs(nlp.vocab):
                # Ensure we pull POS tags correctly
                tags = [token.pos_ for token in doc]  # Extract POS tags from doc
                data.append(Example.from_dict(doc, {"tags": tags}))
    return data

# Load the training and testing data
train_data = load_data(train_data_path)
test_data = load_data(test_data_path)

# Step 3: Set up the training pipeline
if "tagger" not in nlp.pipe_names:
    tagger = nlp.add_pipe("tagger", last=True)

# Step 4: Add POS tags to the tagger
for example in train_data:
    for token in example.reference:
        tagger.add_label(token.pos_)

representative_batch = train_data[:10]  # You can select a small batch (e.g., first 10 examples)

tagger.add_label('X')
# Initialize the tagger with the representative batch
tagger.initialize(lambda: iter(representative_batch))


# Step 6: Initialize the optimizer
optimizer = nlp.begin_training()
# Step 6: Training loop
for epoch in range(10):  # Number of epochs, adjust as necessary
    random.shuffle(train_data)
    losses = {}

    # Iterate over minibatches of the training data
    for batch in minibatch(train_data, size=compounding(4.0, 32.0, 1.001)):
        # Update the model with each batch
        for example in batch:
            nlp.update([example], losses=losses)

    print(f"Epoch {epoch} - Losses: {losses}")

# Step 7: Evaluate the model on the test data
correct = 0
total = 0

# Iterate over the test data
for example in test_data:
    doc = example.predicted
    for token, correct_tag in zip(doc, example.reference):
        print(f"predicted: {token.pos_}, correct: {correct_tag.tag_}")
        if token.pos_ == correct_tag.tag_:
            correct += 1
        total += 1

accuracy = correct / total if total else 0
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 8: Save the trained model
nlp.to_disk("./trained_model")  # Replace with the directory where you want to save the model
print("Model saved to /trained_model")