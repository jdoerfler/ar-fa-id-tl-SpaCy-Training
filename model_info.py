import spacy

models = ['tagalog/output/tl_dep_news_sm', 'persian/output/fa_dep_web_sm', 
          'indonesian/output/id_dep_web_sm', 'arabic/output/ar_dep_web_sm']


import spacy
from spacy.tokens import DocBin

# Load the language model (replace 'xx' with the actual language code you're working with)
nlp = spacy.blank('xx')  # Use a blank model or the actual language model

# Load the .spacy file (DocBin format)
doc_bin = DocBin().from_disk("./arabic/train/ar_padt-ud-train.spacy")

# # Iterate over the documents in the DocBin
# for doc in doc_bin.get_docs(nlp.vocab):
#     print(f"Sentence: {[token.text for token in doc]}")  # Print tokens in the sentence
#     print(f"POS tags: {[token.pos_ for token in doc]}")  # Print POS tags for each token
#     break  # Stop after printing the first document

nlp = spacy.load('arabic/output/model-best')
# Inspect a few sentences from the training data
doc = nlp("برلين ترفض حصول شركة اميركية على رخصة تصنيع دبابة ليوبارد الالماني")
for token in doc:
    print(token.text, token.pos_, token.tag_)