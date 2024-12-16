import spacy

models = ['tagalog/output/tl_dep_news_sm', 'persian/output/fa_dep_web_sm', 
          'indonesian/output/id_dep_web_sm', 'arabic/output/ar_dep_web_sm']

for model in models:
    nlp = spacy.load(model)
    # Get the size of the word vector table
    
    #### test here ####