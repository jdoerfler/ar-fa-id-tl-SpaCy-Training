import spacy

nlp = spacy.load("trained_model")
doc = nlp("نظامی می‌گوید که در سال ۵۱۰ تربت او را زیارت کرده.")
for token in doc:
    token.pos_ = token.tag_
    print(token)
    print(f"Tag: {token.pos_}")
    print("-" * 40)