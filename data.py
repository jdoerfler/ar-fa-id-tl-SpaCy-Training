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
    
    # Initialize lists to store different types of features
    upos_types = []
    xpos_types = []
    dep_types = []
    unique_feats = set()  # Set to store unique morphological feature combinations

    # Loop through all CoNLL files and parse them
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            conllu_data = f.read()

        text = strip_multiword_tokens_from_conllu(filepath)
        sentences = parse(text)

        # Prepare a list to store the training examples
        training_data = []

        # Unique tags and dependency relations for the current sentence
        unique_upos_tags = set()
        unique_xpos_tags = set()
        unique_dep_labels = set()

        for sentence in sentences:
            words = []
            tags = []
            xpos_tags = []
            heads = []
            deps = []
            lemmas = []
            feats = []
            spaces = []
            sent_starts = []

            # Process each token in the sentence
            for token in sentence:
                word = token['form']
                upos_tag = token['upostag']
                xpos_tag = token['xpostag']
                dep_rel = token['deprel']
                lemma = token['lemma']
                feat = token['feats']
                
                # Process morphological features
                if feat:
                    # Check if feat is a dictionary or a string
                    if isinstance(feat, str):
                        feat_dict = dict(f.split('=') for f in feat.split('|'))
                    elif isinstance(feat, dict):
                        feat_dict = feat
                    else:
                        feat_dict = {}

                    # Add the feature dictionary to the set of unique morphological features
                    unique_feats.add(frozenset(feat_dict.items()))  # Use frozenset to store unique combinations

                head = int(token['head'])  # Head is 1-based index
                try:
                    space_after = (token['misc']['SpaceAfter'] == 'Yes')
                except:
                    if token['form'] in ['.', '?', '!']:
                        space_after = False
                    else:
                        space_after = True
                sent_start = (token['id'] == 1)

                # Collect the unique POS tags and dependency labels
                unique_upos_tags.add(upos_tag)
                unique_xpos_tags.add(xpos_tag)
                unique_dep_labels.add(dep_rel)

                # Store token-level information for building the Example
                words.append(word)
                tags.append(upos_tag)
                xpos_tags.append(xpos_tag)
                heads.append(fix_head_index(head, token['id']))
                deps.append(dep_rel)
                lemmas.append(lemma)
                feats.append(feat)
                spaces.append(space_after)
                sent_starts.append(sent_start)

            # Create a spaCy Doc object
            doc = nlp.make_doc(' '.join(words))  # Create a doc from the sentence
            annotations = {
                "words": words,
                "pos": tags,
                "lemmas": lemmas,
                "heads": heads,
                "deps": deps,
                "spaces": spaces,
                "sent_starts": sent_starts
            }

            example = Example.from_dict(doc, annotations)
            training_data.append(example)

        # Add the training data to the combined dataset
        combined_conllu_data.extend(training_data)

        # Add unique tags and dependency labels
        upos_types.extend(unique_upos_tags)
        xpos_types.extend(unique_xpos_tags)
        dep_types.extend(unique_dep_labels)

    # Print out summary
    print("=" * 60)
    print(f"Spacy-friendly data created with \033[1;92m{len(combined_conllu_data)}\033[0m samples!")
    print("=" * 60)

    # Return the combined dataset and the unique features for the morphologizer
    return combined_conllu_data, list(set(upos_types)), list(set(xpos_types)), list(set(dep_types)), [dict(feat) for feat in unique_feats]

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
    languages = ['arabic', 'indonesian','tagalog','persian']
    lang_codes = ['ar','id','tl','fa']
    for language, code in zip(languages, lang_codes):
        for test_train in ['test','train']:
            get_conllu_data('tagalog/train', spacy.blank(f'{code}'))