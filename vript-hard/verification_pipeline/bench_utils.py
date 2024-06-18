import tqdm
import datasets
import re
import pandas as pd

gt_json_files = {
    'HAL': 'HAL_annotations.jsonl',
    'RR': 'RR_annotations.jsonl',
    'ERO': 'ERO_annotations.jsonl',
}

gt_answer_keys = {
    'HAL': 'content',
    'RR-open': 'open_answer',
    'RR-multiple': 'multiple_choice_answer',
    'ERO': 'answer',
}

class HallucinationScorer():
    def __init__(self):
        import spacy
        from sentence_transformers import SentenceTransformer, util

        self.util = util
        self.nlp = spacy.load('en_core_web_lg')
        self.sentence_model = SentenceTransformer("all-mpnet-base-v2")

    def extract_verbs_and_nouns(self, text):
        doc = self.nlp(text)
        words = []
        words.extend([token.text for token in doc if token.pos_ == 'VERB'])
        words.extend([token.text for token in doc if token.pos_ == 'NOUN'])
        words.extend([token.text for token in doc if token.pos_ == 'PROPN'])

        # lemmatize the words
        words = [self.nlp(word)[0].lemma_ for word in words]
        words = list(set(words))

        return words
    
    def remove_stopword_and_stemming(self, words):
        removed = [word for word in words if word not in self.nlp.Defaults.stop_words]
        stemmed = [self.nlp(word)[0].lemma_ for word in removed]
        
        return stemmed
    
    def compute_precision_recall_score(self, gt, pred):
        
        gt_words = []
        for gt_ in gt:
            gt_words.extend(self.extract_verbs_and_nouns(gt_))
        pred_words = self.extract_verbs_and_nouns(pred)

        # remove stopwords
        gt_words = self.remove_stopword_and_stemming(gt_words)
        pred_words = self.remove_stopword_and_stemming(pred_words)
        
        gt_words = list(set(gt_words))
        pred_words = list(set(pred_words))

        # compute word embeddings
        gt_embedding = self.sentence_model.encode(gt_words, convert_to_tensor=True)
        pred_embedding = self.sentence_model.encode(pred_words, convert_to_tensor=True)

        # compute cosine similarity
        precision_num = 0
        for pred_embed in pred_embedding:
            cosine_scores = self.util.cos_sim(gt_embedding, pred_embed).squeeze(-1)

            # if the cosine similarity is greater than 0.5, we consider it as a match
            if sum(cosine_scores > 0.5) > 0:
                precision_num += 1
                
        precision_score = precision_num / len(pred_words)
        recall_score = precision_num / len(gt_words)

        return precision_score, recall_score
    
    def compute_f1_score(self, precision, recall):
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)

    def compute_score(self, gts, preds):
        precision_scores = []
        recall_scores = []
        f1_scores = []
        bar = tqdm.tqdm(gts.keys(), desc="Hallucination Evaluation")
        for gt, pred in zip(gts.values(), preds.values()):
        # def compute_score_(gt, pred):
            precision, recall = self.compute_precision_recall_score(gt, pred[0])
            f1_score = self.compute_f1_score(precision, recall)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1_score)
            bar.update(1)
            
        # joblib.Parallel(n_jobs=8, backend="threading")(joblib.delayed(compute_score_)(gt, pred) for gt, pred in tqdm.tqdm(zip(gts.values(), preds.values()), total=len(gts)))

        return f1_scores, (precision_scores, recall_scores)

class HALScorer():
    def __init__(self, traditional=True, cache_dir=None):
        self.scorers = []
        if traditional:
            from pycocoevalcap.cider.cider import Cider
            from pycocoevalcap.bleu.bleu import Bleu
            # from pycocoevalcap.meteor.meteor import Meteor
            from pycocoevalcap.rouge.rouge import Rouge
            # from pycocoevalcap.spice.spice import Spice

            self.traditional_scorers = [
                (Cider(), "CIDEr"),
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                # (Meteor(), "METEOR"),
                (Rouge(), "ROUGE_L"),
                # (Spice(), "SPICE"),
            ]
            self.scorers.extend(self.traditional_scorers)
        else:
            self.Hallucination_scorer = (HallucinationScorer(), "Hallucination")
            self.scorers.append(self.Hallucination_scorer)

        gt_data_file = {"test": gt_json_files["HAL"]}
        gt_dataset = datasets.load_dataset(f"mutonix/Vript-HAL", data_files=gt_data_file, cache_dir=cache_dir)['test']
        self.gt_answers_dict = {gt['clip_id']: gt[gt_answer_keys["HAL"]] for gt in gt_dataset}
    
    def compute_scores(self, predictions, output_file=None):
        # assert len(self.gt_answers_dict) == len(predictions), "The number of predictions does not match the number of groud truth answers."

        gts = {}
        preds = {}
        for i, pred in predictions.iterrows():
            clip_id = pred['id']
            if 'gt' in pred:
                gts[i] = [pred['gt']]
            else:
                gts[i] = [self.gt_answers_dict[clip_id]] if isinstance(self.gt_answers_dict[clip_id], str) else self.gt_answers_dict[clip_id]
            preds[i] = [pred['pred']]

        total_scores = {}
        for scorer, method in self.scorers:
            # print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, preds)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print(f"Vript-HAL [{m}]: {sc}")
                total_scores["Bleu"] = score
            elif method == "Hallucination":
                predictions = pd.concat([
                    predictions,
                    pd.Series(score, name="f1_score"),
                    pd.Series(scores[0], name="precision"),
                    pd.Series(scores[1], name="recall")
                ], axis=1)
                if output_file is not None:
                    predictions.to_csv(output_file, index=False)
                
                print(f"Vript-HAL [Hallucination]  F1:  {sum(score) / len(score) * 100:.2f}")
                print(f"Vript-HAL [Hallucination]  Precision:  {sum(scores[0]) / len(scores[0]) * 100:.2f}")
                print(f"Vript-HAL [Hallucination]  Recall:  {sum(scores[1]) / len(scores[1]) * 100:.2f}\n")
            else:
                print(f"Vript-HAL [{method}]: {score}")
                total_scores[method] = score

class RRScorer():
    def __init__(self, task_type='full', cache_dir=None):
        assert task_type in ['full', 'clip'], "task_type should be either 'full' or 'clip'"
        self.task_type = task_type
        
        gt_data_file = {"test": gt_json_files["RR"]}
        gt_dataset = datasets.load_dataset(f"mutonix/Vript-RR", data_files=gt_data_file, cache_dir=cache_dir)['test']      
        # self.gt_answers_dict = {gt['clip_id']: gt[gt_answer_keys["RR-multiple"]] for gt in gt_dataset}
        self.gt_answers_dict = {}
        for gt in gt_dataset:
            choices = gt['multiple_choice']
            answer = gt['multiple_choice_answer']
            # (A) (B) (C) (D)
            self.gt_answers_dict[gt['clip_id']] = f"({chr(65 + choices.index(answer))})"

    def compute_scores(self, predictions, output_file=None):
        # assert len(self.gt_answers_dict) == len(predictions), "The number of predictions does not match the number of groud truth answers."

        gts = {}
        preds = {}

        video_ids = []
        for i, pred in predictions.iterrows():
            if pd.isna(pred['pred']):
                continue
            
            video_ids.append(pred['id'])
            if 'gt' in pred:
                gts[i] = re.search(r'\([A-Z]\)', pred['gt']).group(0)
            else:
                clip_id = pred['id']
                gts[i] = self.gt_answers_dict[clip_id]
            
            if re.search(r'\([A-Z]\)', pred['pred']):
                preds[i] = re.search(r'\([A-Z]\)', pred['pred']).group(0)
            else:
                preds[i] = None
                        

        tf = []
        for k in gts.keys():
            # if preds[k] is not None:
            tf.append(gts[k] == preds[k])
            # else:
            #     pass
        
        acc = round(sum(tf) / len(tf) * 100, 2) 
        
        if acc != 0:
            predictions = pd.concat([
                predictions,
                pd.Series(tf, name="correct"),
            ], axis=1)
            if output_file is not None:
                predictions.to_csv(output_file, index=False)

        print(f'Vript-RR [{self.task_type} multiple-choice] acc: {sum(tf)}/{len(tf)} {acc}\n')


class EROScorer():
    def __init__(self, cache_dir=None):

        gt_data_file = {"test": gt_json_files["ERO"]}
        gt_dataset = datasets.load_dataset(f"mutonix/Vript-ERO", data_files=gt_data_file, cache_dir=cache_dir)['test']
        self.gt_answers_dict = {gt['video_id']: gt[gt_answer_keys["ERO"]] for gt in gt_dataset}

    def compute_scores(self, predictions, output_file=None):
        gts = {}
        preds = {}

        video_ids = []
        for i, pred in predictions.iterrows():
            if pd.isna(pred['pred']):
                continue
            
            video_ids.append(pred['id'])
            if 'gt' in pred:
                gts[i] = re.findall(r'([A-Z])', pred['gt'])
            else:
                clip_id = pred['id']
                gts[i] = self.gt_answers_dict[clip_id]

            preds[i] = re.findall(r'Scene \(([A-Z])\)', pred['pred'])[:3]
            if len(preds[i]) < 3:
                preds[i] = re.findall(r'([A-Z])', pred['pred'])[:3]

        tf_top1 = []
        tf_top2 = []
        tf_top3 = []
        for k in gts.keys():
            if preds[k] is not None and len(preds[k]) == 3:
                A = gts[k][0] == preds[k][0]
                B = gts[k][1] == preds[k][1]
                C = gts[k][2] == preds[k][2]
                tf_top1.append(A or B or C)
                tf_top2.append((A and B) or (A and C) or (B and C))
                tf_top3.append(A and B and C)         
            else:
                tf_top1.append(False)
                tf_top2.append(False)
                tf_top3.append(False)
        
        acc_top1 = round(sum(tf_top1) / len(tf_top1) * 100, 2)
        acc_top2 = round(sum(tf_top2) / len(tf_top2) * 100, 2)
        acc_top3 = round(sum(tf_top3) / len(tf_top3) * 100, 2)
    
        acc = [acc_top1, acc_top2, acc_top3]
        tf_tops = [tf_top1, tf_top2, tf_top3]
        
        for i, acc_ in enumerate(acc):
            if acc_ != 0:
                print(f'Vript-ERO @{i + 1} acc: {sum(tf_tops[i])}/{len(tf_tops[i])} {acc_}')
                predictions = pd.concat([
                    pd.Series(tf_tops[i], name=f"top{i + 1}"),
                    predictions,
                ], axis=1)
                if output_file is not None:
                    predictions.to_csv(output_file, index=False)
        