# python -m spacy convert arabic\train\ar_padt-ud-train.conllu arabic\train\ -n 10 --converter conllu
# python -m spacy train config.cfg --output .\{lang}_boot_sm --paths.train .\train
# python -m spacy evaluate indonesian/id_pos_sm/model-last indonesian/test/