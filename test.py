import spacy
import data
from spacy.training.example import Example

if __name__ == "__main__":
    languages = ['tagalog'] #['arabic','indonesian','persian','tagalog']
    lang_codes = ['tl'] #['ar','id','fa','tl']
    spacy.prefer_gpu()

    for language, lang_code in zip(languages, lang_codes):
        nlp = spacy.load(f"{language}/output/{lang_code}_dep_web_sm")
        test_data, _, _ = data.get_conllu_data(f"{language}/test", nlp)
        print(f"Loaded model for {language} ({lang_code})")


        # Evaluate the model on the test data
        # eval_examples = [Example.from_dict(nlp.make_doc(ex.text), ex.reference) for ex in test_data]
        eval_examples = []
        for example in test_data:
            # The example already has the annotations (i.e., 'reference' field) and predictions (in 'predicted' field)
            eval_examples.append(example)
        scores = nlp.evaluate(eval_examples)
        

        first_example = eval_examples[0]

        # True values (reference)
        doc_ref = first_example.reference  # True values (reference)
        doc_pred = nlp(first_example.text)  # Get predictions from the model
        total_preds = len(doc_pred)
        correct_preds = 0
        for ref_token, pred_token in zip(doc_ref, doc_pred):
            print(f"Reference: \tToken: \t{ref_token.text}\t Head: {ref_token.head.text}\t Dep: {ref_token.dep_}\t POS: {ref_token.pos_}")
            print(f"Predicted: \tToken: {pred_token.text}\t Head: {pred_token.head.text}]\t Dep: {pred_token.dep_}\t POS: {pred_token.tag_}\n")
            if ref_token.pos_ == pred_token.tag_:
                correct_preds += 1
        

        # Print evaluation results
        print(f"Evaluation results for {language} ({lang_code}):")
        # print(f"  - Precision: {scores['token_p']:.4f}")
        # print(f"  - Recall: {scores['token_r']:.4f}")
        # print(f"  - F1 Score: {scores['token_f']:.4f}")
        print(f"  - Tagger accuracy: {correct_preds/total_preds:.4f}")
        #print(f"  - Tagger accuracy: {scores['tag_acc']:.4f}")
        print(f"  - Dep head accuracy: {scores['dep_uas']:.4f}")
        print(f"  - Dep label accuracy: {scores['dep_las']:.4f}")
       