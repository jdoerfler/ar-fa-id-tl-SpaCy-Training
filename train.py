import spacy
import time
import random
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import data  
from tqdm import tqdm

if __name__ == "__main__":
    languages = ['tagalog']#, 'persian']  # ['arabic']#
    lang_codes = ['tl', 'fa']  # ar

    # if spacy.prefer_gpu():
    #     print("Running with \033[1;92mGPU\033[0m")
    # else:
    #     print("Running with \033[1;31mNO GPU\033[0m")

    for i, language in enumerate(languages):
        start_time = time.time()
        
        # Initialize the blank spacy model
        nlp = spacy.blank(f"{lang_codes[i]}")

        # Add the 'tok2vec', 'tagger', and 'parser' components if not already present
        if 'tok2vec' not in nlp.pipe_names:
            nlp.add_pipe('tok2vec', first=True)  # Add as the first component
        
        if 'tagger' not in nlp.pipe_names:
            nlp.add_pipe('tagger', last=True)  # Add POS tagger as the last component
        
        if 'parser' not in nlp.pipe_names:
            nlp.add_pipe('parser', before='tagger')  # Add the parser before the tagger component

        # Load the training data
        train_data, pos_tags, dep_labels = data.get_conllu_data(f"{language}/train", nlp)
        
        #print(train_data[:1])
        
        # Get the tagger and parser
        tagger = nlp.get_pipe("tagger")
        parser = nlp.get_pipe("parser")
        
        # Add labels to the tagger
        for tag in pos_tags:
            tagger.add_label(tag)
        
        # Add labels to the parser
        for dep_label in dep_labels:
            parser.add_label(dep_label)

        # Start training the model
        optimizer = nlp.begin_training()
        n_iter = 1  # Set the number of iterations for training

        for epoch in range(n_iter):
            epoch_start_time = time.time()
            random.shuffle(train_data)  # Shuffle data for each epoch
            losses = {}
            
            # Create minibatches and train the model
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            
            batch_size = 1460
            for j, batch in tqdm(enumerate(batches), total=batch_size, ncols=100):
                batch_size *= 1.001
                for example in batch:
                    nlp.update([example], losses=losses)

            epoch_end_time = time.time()
            print(f"Epoch {epoch + 1}/{n_iter}, Losses: {losses}, Time: {epoch_end_time - epoch_start_time:.2f} seconds")

        # Save the trained model
        model_output_path = f"{language}/output/{lang_codes[i]}_dep_web_sm"
        nlp.to_disk(model_output_path)
        
        end_time = time.time()
        print("@" * 60)
        print(f"     Time elapsed for {language}: {end_time - start_time:.2f} seconds")
        print(f" ________________________________________________________ ")
        print(f"|                                                        | ")
        print(f"|      Training complete and model saved to disk.        |")
        print(f"|________________________________________________________|")
        print("@" * 60)
