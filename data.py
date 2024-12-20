import os
import re
from spacy.training.example import Example 
from conllu import parse
from spacy.tokens import Doc

def get_conllu_filepaths_from_directory(directory: str) -> list[str]:
    filepaths = []
    for filename in os.listdir(directory):
        if filename.endswith('.conllu'):
            filepath = os.path.join(directory, filename)
            filepaths.append(filepath)
    return filepaths

def strip_multiword_tokens_from_conllu(filepath: str): # removes multiword tokens, which may be incompatible with SpaCy or I just couldn't figure it out
    
    # Define the regex pattern to match lines with '[0-9]+-[0-9]+'
    pattern = r"^[0-9]{1,3}[-.][0-9]{1,3}\t"
    
    # Open the CoNLL file and read its content
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Filter lines that do not match the pattern
    cleaned_lines = [line for line in lines if not re.match(pattern, line.strip())]
    
    # each file's last entry is an empty extra line, which breaks spacy.conll
    if lines[-2:] == ['\n', '\n']:
        lines.pop[-1]
    print("=" * 60)
    print(f"Finished stripping multiword tokens from {filepath}")
    return "".join(cleaned_lines)


def get_conllu_data(directory: str, nlp):
    combined_conllu_data = []
    filepaths = get_conllu_filepaths_from_directory(directory)
    # conll_nlp = spacy_conll.ConllParser(spacy_conll.init_parser('xx_ent_wiki_sm', 'spacy'))
    
    pos_types = []
    dep_types = []
    # Loop through all CoNLL files and parse them
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            conllu_data = f.read()

        
        text = strip_multiword_tokens_from_conllu(filepath)
        # Parse the CONLL-U data
        sentences = parse(text)
        
        # Prepare a list to store the training examples
        training_data = []

        unique_pos_tags = []
        unique_dep_labels = []

        for sentence in sentences:
            # Extract the tokens and their annotations for this sentence
            words = []
            tags = []
            heads = []
            deps = []
            spaces = []
            sent_starts = []

            # Process each token in the sentence
            for token in sentence:
                # Get the text (word), POS, and dependency relation
                word = token['form']
                pos_tag = token['upostag']
                dep_rel = token['deprel']
                head = int(token['head'])  # Head is 1-based index
                try:
                    space_after = (token['misc']['SpaceAfter'] == 'Yes')
                except:
                    if token['form'] in ['.','?','!']:
                        space_after = False
                    else:
                        space_after = True
                if token['id'] == 1:
                    sent_start = True
                else:
                    sent_start = False

                if pos_tag not in unique_pos_tags:
                    unique_pos_tags.append(pos_tag)
                if dep_rel not in unique_dep_labels:
                    unique_dep_labels.append(dep_rel)

                spacy_pos_list = [
                    "ADJ",   # Adjective
                    "ADP",   # Adposition
                    "ADV",   # Adverb
                    "AUX",   # Auxiliary verb
                    "CCONJ", # Coordinating conjunction
                    "DET",   # Determiner
                    "INTJ",  # Interjection
                    "NOUN",  # Noun
                    "NUM",   # Numeral
                    "PART",  # Particle
                    "PRON",  # Pronoun
                    "PROPN", # Proper noun
                    "PUNCT", # Punctuation
                    "SCONJ", # Subordinating conjunction
                    "SYM",   # Symbol
                    "VERB",  # Verb
                    "X"    ] # Other
                words.append(word)
                
                if pos_tag in spacy_pos_list:
                    tags.append(pos_tag) # ensure all tags are good to go - was having troubles getting tagger component to predict in the .pos_ attribute instead of the .tag_ attribute
                else:
                    tags.append('X') # default to 'other' if there's some issue

                fixed_head = fix_head_index(head, token['id'])
                heads.append(fixed_head)
                deps.append(dep_rel)
                spaces.append(space_after)
                sent_starts.append(sent_start)

            # fix heads here
            
            # Create a doc object using the words (this will use spaCy's tokenizer)
            doc = nlp.make_doc(' '.join(words))  # Create a doc from the sentence
            annotations = {
                "words": words,   # The words (tokens)
                "pos": tags,     # The POS tags
                "heads": heads ,   # The head (parent) indices
                "deps": deps,     # Dependency relations
                "spaces": spaces, 
                "sent_starts": sent_starts
            }


            example = Example.from_dict(doc, annotations)
            
            training_data.append(example)
        
            
        combined_conllu_data.extend(training_data)
        for tag, label in zip(unique_pos_tags, unique_dep_labels):
            if tag not in pos_types:
                pos_types.append(tag)
            if label not in dep_types:
                dep_types.append(label)
    
    print("=" * 60)
    print(f"Spacy-friendly data created with \033[1;92m{len(combined_conllu_data)}\033[0m samples!")
    print("=" * 60)
    return combined_conllu_data, pos_types, dep_types

def fix_head_index(head, token_id):
    """
    Fixes the head indices in the annotation data by ensuring they are ABSOLUTE and HEAD is not 0, which is CoNLL-U standard.
    """
    if head == 0:
        return token_id - 1
    else:
        return head - 1
    


if __name__ == "__main__":
    import spacy
    get_conllu_data('tagalog/train', spacy.blank('tl'))