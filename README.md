### python -m spacy convert {language}\{train_or_test}\{lang_code}_{dataset}-ud-{train_or_test}.conllu {language}\{train_or_test}\ -n 10 --converter conllu
### python -m spacy train config.cfg --output ./output --gpu-id 0
### python -m spacy evaluate {language}/output/{model_name} {language}/test/ --gpu-id 0


Indonesian:
Trained on [id_csui-ud-train.conllu, id_gsd-ud-train.conllu]
Tested on [id_csui-ud-test.conllu, id_gsd-ud-test.conllu]
SpaCy evaluate stats:
TOK     98.08
TAG     80.32
SPEED   25275

Arabic:
Trained on [ar_padt-ud-train.conllu]
Tested on [ajp_madar-ud-test.conllu, ar_padt-ud-test.conllu]
SpaCy evaluate stats:
TOK     84.00
TAG     40.63
SPEED   42311

PERSIAN:
Trained on [fa_perdt-ud-train.conllu, fa_seraji-ud-train.conllu]
Tested on [fa_perdt-ud-test.conllu, fa_seraji-ud-test.conllu]
SpaCy evaluate stats: 
TOK     99.96
TAG     54.50
SPEED   19211

TAGALOG: 
Trained on tl_newscrawl-ud-train.conllu
Tested on [tl_newscrawl-us-test.conllu, tl_trg-ud-test.conllu, tl_ugnayan-ud-test.conllu]
SpaCy evaluate stats:
TOK     99.64
TAG     93.68
SPEED   44800