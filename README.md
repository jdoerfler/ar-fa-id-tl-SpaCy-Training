For all .conllu files, ran convert_xpos_to_upos to let tagger train off just part of speech
### python -m spacy convert {language}\{train_or_test}\{lang_code}_{dataset}-ud-{train_or_test}.conllu {language}\{train_or_test}\ -n 10 --converter conllu
### python -m spacy train config.cfg --output ./output --gpu-id 0
### python -m spacy evaluate {language}/output/{model_name} {language}/test/ --gpu-id 0


Indonesian:
Trained on [id_csui-ud-train.conllu, id_gsd-ud-train.conllu]
Tested on [id_csui-ud-test.conllu, id_gsd-ud-test.conllu]
https://huggingface.co/jdoerfler/SpaCy-id_dep_web_sm

Arabic:
Trained on [ar_padt-ud-train.conllu]
Tested on [ajp_madar-ud-test.conllu, ar_padt-ud-test.conllu]
https://huggingface.co/jdoerfler/SpaCy-ar_dep_web_sm

PERSIAN:
Trained on [fa_perdt-ud-train.conllu, fa_seraji-ud-train.conllu]
Tested on [fa_perdt-ud-test.conllu, fa_seraji-ud-test.conllu]
https://huggingface.co/jdoerfler/SpaCy-fa_dep_web_sm

TAGALOG: 
Trained on tl_newscrawl-ud-train.conllu
Tested on [tl_newscrawl-us-test.conllu, tl_trg-ud-test.conllu, tl_ugnayan-ud-test.conllu]
https://huggingface.co/jdoerfler/SpaCy-tl_dep_web_sm
