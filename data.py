import os
import re
from spacy.training.example import Example 
import spacy_conll
from conllu import parse

def get_conllu_filepaths_from_directory(directory: str) -> list[str]:
    filepaths = []
    for filename in os.listdir(directory):
        if filename.endswith('.conllu'):
            filepath = os.path.join(directory, filename)
            filepaths.append(filepath)
    return filepaths

def strip_multiword_tokens_from_conllu(filepath: str):
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

def get_conll_data(directory: str, nlp):
    conll_data = []
    filepaths = get_conllu_filepaths_from_directory(directory)
    conll_nlp = spacy_conll.ConllParser(spacy_conll.init_parser('xx_ent_wiki_sm', 'spacy'))
    
    pos_types = []
    dep_types = []
    # Loop through all CoNLL files and parse them
    for filepath in filepaths:
        #doc = conll_nlp.parse_conll_file_as_spacy(filepath, input_encoding='utf-8')
        text = strip_multiword_tokens_from_conllu(filepath)
        doc = conll_nlp.parse_conll_text_as_spacy(text)
        unique_pos_tags = []
        unique_dep_labels = []
        # Create training examples from the parsed doc
        for sent in doc.sents:
            words = []
            pos_tags = []
            for token in sent:
                words.append(token.text)
                pos_tags.append(token.pos_)
                dep_labels = [token.dep_ for token in sent]

                if token.pos_ not in unique_pos_tags:
                    unique_pos_tags.append(token.pos_)
                if token.dep_ not in unique_dep_labels:
                    unique_dep_labels.append(token.dep_)

            # words = [token.text for token in sent]
            # pos_tags = [token.pos_ for token in sent]
            annotations = {
                    'tags': pos_tags,
                    'deps': dep_labels
                }
            try: 
                example = Example.from_dict(nlp.make_doc(" ".join(words)), annotations)
                conll_data.append(example)
            except: pass
        
        for tag, label in zip(unique_pos_tags, unique_dep_labels):
            if tag not in pos_types:
                pos_types.append(tag)
            if label not in dep_types:
                dep_types.append(label)

    print("=" * 60)
    print(f"Spacy-friendly data created with \033[1;92m{len(conll_data)}\033[0m samples!")
    print("=" * 60)
    return conll_data, pos_types, dep_types

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

                # Append to the lists
                words.append(word)
                tags.append(pos_tag)
                fixed_head = fix_head_index(head, token['id'])
                heads.append(fixed_head)
                deps.append(dep_rel)
                spaces.append(space_after)
                sent_starts.append(sent_start)

            # fix heads here
            
            # Create a doc object using the words (this will use spaCy's tokenizer)
            doc = nlp.make_doc(' '.join(words))  # Create a doc from the sentence
            for token in doc:
                print(token)
            annotations = {
                "words": words,   # The words (tokens)
                "tags": tags,     # The POS tags
                "heads": heads ,   # The head (parent) indices
                "deps": deps,     # Dependency relations
                "spaces": spaces, 
                "sent_starts": sent_starts
            }
            # Create the Example object with the annotations
            example = Example.from_dict(doc, annotations)
            
            # Append the example to the training data list
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
    Fixes the head indices in the annotation data by ensuring they are relative to the token list.
    """
    # head is absolute, needs to be relative
    # heads = [3,1,0,3,3,3]
    # idx = [2,1,-1,2,2,2]
    if head == 0:
        return 0
    else:
        return head - token_id
    


if __name__ == "__main__":
    import spacy
    get_conllu_data('tagalog/train', spacy.blank('tl'))