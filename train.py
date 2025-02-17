import spacy
import time
import random
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import data  
from tqdm import tqdm
import os

def train_language(language, lang_code):
    start_time = time.time()
        
    # Ensure spaCy uses the GPU if available
    spacy.require_gpu()  # Ensure that spaCy uses the GPU if available

    # Initialize the blank spacy model
    nlp = spacy.load("xx_ent_wiki_sm")

    # Add components in correct order with proper configuration
    if 'tok2vec' not in nlp.pipe_names:
        nlp.add_pipe('tok2vec', first=True)
        
    if 'morphologizer' not in nlp.pipe_names:
        nlp.add_pipe('morphologizer')
    
    if 'parser' not in nlp.pipe_names:
        nlp.add_pipe('parser')
    
    if 'tagger' not in nlp.pipe_names:
        nlp.add_pipe('tagger')
    
    # Load the training data
    train_data, upos_tags, xpos_tags, dep_labels, features = data.get_conllu_data(f"{language}/train", nlp)
    
    # Get the tagger and parser
    tagger = nlp.get_pipe("tagger")
    parser = nlp.get_pipe("parser")
    morphologizer = nlp.get_pipe('morphologizer')
    
    # Add UPOS tags to morphologizer
    for tag in upos_tags:
        morphologizer.add_label(f"POS={tag}")

    for feature_dict in features:
        for feature, value in feature_dict.items():
            label = f"{feature}={value}"
            morphologizer.add_label(label)
    
    for tag in upos_tags:
        tagger.add_label(tag)
    
    # Add labels to the parser
    for dep_label in dep_labels:
        parser.add_label(dep_label)

    # Start training the model
    optimizer = nlp.begin_training()
    n_iter = 10  # Set the number of iterations for training

    for epoch in range(n_iter):
        epoch_start_time = time.time()
        random.shuffle(train_data)  # Shuffle data for each epoch
        losses = {}
        
        # Create minibatches and train the model
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
        k = 0
        for j, batch in enumerate(batches):
            for example in batch:
                nlp.update([example], losses=losses)
                k += 1
                if k % 500 == 0:
                    print(f'{lang_code} at example {k} in batch {j}')

        epoch_end_time = time.time()
        print(f"{language}: Epoch {epoch + 1}/{n_iter}, Losses: {losses}, Time: {epoch_end_time - epoch_start_time:.2f} seconds")

    # Save the trained model
    model_output_path = f"{language}/output/{lang_code}_dep_web_sm"
    nlp.to_disk(model_output_path)
    
    end_time = time.time()
    print("@" * 60)
    print(f"     Time elapsed for {language}: {end_time - start_time:.2f} seconds")
    print(f" ________________________________________________________ ")
    print(f"|                                                        | ")
    print(f"|      Training complete and model saved to disk.        |")
    print(f"|________________________________________________________|")
    print("@" * 60)

if __name__ == "__main__":
    # Define languages and their codes
    language_pairs = [
        #('tagalog', 'tl'),
        ('persian', 'fa'),
        #('arabic', 'ar'),
        #('indonesian','id')
    ]
    
    # Start training for all languages sequentially
    for language, lang_code in language_pairs:
        train_language(language, lang_code)

    print("\nTraining completed for all languages.")