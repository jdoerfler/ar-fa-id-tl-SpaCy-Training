import spacy
import data
from spacy.training.example import Example

if __name__ == "__main__":
    languages = ['arabic','indonesian','persian','tagalog']
    lang_codes = ['ar','id','fa','tl']
    spacy.prefer_gpu()

    for language, lang_code in zip(languages, lang_codes):
        nlp = spacy.load(f"{language}/output/{lang_code}_dep_web_sm")
        test_data, _, _ = data.get_conll_data(f"{language}/test", nlp)
        print(f"Loaded model for {language} ({lang_code})")


        # Evaluate the model on the test data
        # eval_examples = [Example.from_dict(nlp.make_doc(ex.text), ex.reference) for ex in test_data]
        eval_examples = []
        for example in test_data:
            # The example already has the annotations (i.e., 'reference' field) and predictions (in 'predicted' field)
            eval_examples.append(example)
        scores = nlp.evaluate(eval_examples)

        # Print evaluation results
        print(f"Evaluation results for {language} ({lang_code}):")
        # print(f"  - Precision: {scores['token_p']:.4f}")
        # print(f"  - Recall: {scores['token_r']:.4f}")
        # print(f"  - F1 Score: {scores['token_f']:.4f}")
        print(f"  - Tagger accuracy: {scores['tag_acc']:.4f}")
        print(f"  - Dep head accuracy: {scores['dep_uas']:.4f}")
        print(f"  - Dep label accuracy: {scores['dep_las']:.4f}")
       