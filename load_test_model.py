import spacy

nlp = spacy.load("trained_model")
doc = nlp(  "سوريا: تعديل وزاري واسع يشمل 8 حقائب" )
for token in doc:
    token.pos_ = token.tag_
    print(token)
    print(f"Tag: {token.pos_}")
    print("-" * 40)