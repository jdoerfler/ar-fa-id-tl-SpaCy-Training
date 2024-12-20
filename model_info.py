import spacy

# Load a model with a trained Morphologizer (e.g., 'xx_ent_wiki_sm')
nlp = spacy.load('tagalog/output/tl_dep_web_sm')

# Access the Morphologizer component from the pipeline
morphologizer = nlp.get_pipe('morphologizer')

# Get the morphological labels
print(morphologizer.labels)

doc = nlp("She has been playing football.")

# Iterate through the tokens and print both POS tags and morphological features
for token in doc:
    print(f"Token: {token.text}")
    print(f"  POS tag (token.pos_): {token.pos_}")  # Universal POS tag (UPOS)
    print(f"  Morphological features (token.morph): {token.morph}")  # Morphological features
    print("-" * 50)