import spacy
from spacy.util import minibatch, compounding
import random
import data
import time


if __name__ == "__main__":
    languages = ['persian','tagalog'] # ['arabic']#,
    lang_codes = ['fa','tl'] # ar

    for i, language in enumerate(languages):
        spacy.prefer_gpu()
        start_time = time.time()
        # Initialize the spacy model using spacy_conll
        nlp = spacy.blank(f"{lang_codes[i]}")

        # Add the 'tok2vec' and 'tagger' components if not already present
        if 'tok2vec' not in nlp.pipe_names:
            nlp.add_pipe('tok2vec', first=True)  # Add as the first component
        
        if 'tagger' not in nlp.pipe_names:
            nlp.add_pipe('tagger', last=True)  # Add POS tagger as the last component

             # Add the 'parser' component for dependency parsing
        if 'parser' not in nlp.pipe_names:
            nlp.add_pipe('parser', before='tagger')  # Add the parser before the tagger component

        train_data, pos_tags, dep_labels = data.get_conll_data(f"{language}/train", nlp)
        
        tagger = nlp.get_pipe("tagger")
        # Add labels to the tagger
        for tag in pos_tags:
            tagger.add_label(tag)

        parser = nlp.get_pipe("parser")
    
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
            for batch in batches:
                for example in batch:
                    nlp.update([example], losses=losses)
            epoch_end_time = time.time()
            print(f"Epoch {epoch + 1}/{n_iter}, Losses: {losses}, in {epoch_end_time - epoch_start_time} seconds")

        # Save the trained model
        nlp.to_disk(f"{language}/output/{lang_codes[i]}_dep_web_sm")
        print("Training complete and model saved to disk.")
        end_time = time.time()
        print(f"Time elapsed for {language}: {end_time - start_time}")