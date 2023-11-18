import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

id2label = {
    0: 'False',
    1: 'True',
}

label2id = {
    'False': 0,
    'True': 1,
}

num_labels = len(id2label)

class QAEnsembleModel(nn.Module):
    def __init__(self, model_checkpoint, device="cpu"):
        super(QAEnsembleModel, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            use_fast=True,
            cls_token="<s>",
            sep_token="</s>"
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_labels
        ).to(device)

    def forward(self, question, choices, texts, ranking_scores=None):
        if ranking_scores is None:
            ranking_scores = np.ones((len(texts),))

        best_positive_index = 0
        best_positive_score = 0
        all_choices_answers = []
        for idx, c in enumerate(choices):
            if c.split('.', 1)[0] in ["A", "B", "C", "D", "E", "F"]:
                c = c[2:].strip()

            answers = []
            answer_scores = []
            for text, score in zip(texts, ranking_scores):
                prompt = f"<s>{text}</s>{question}</s>{c}</s>"
                print(prompt)
                model_inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=256,
                    truncation=True,
                    return_tensors="pt"
                )
                outputs = self.model(**model_inputs)
                prediction = torch.argmax(outputs[0], axis=1).item()
                _l = outputs[0].detach().numpy()[0] * score
                # {0,1}, score
                answers.append(str(prediction))
                answer_scores.append(_l[prediction])
                confident_score = _l[0] - _l[1]
                if confident_score > best_positive_score:
                    best_positive_score = confident_score
                    best_positive_index = idx
            # find best choices
            best_answers_idx = np.argmax(np.array(answer_scores))
            choice_answer = answers[best_answers_idx]
            all_choices_answers.append(choice_answer)
        
        # do some trick to correct answer, each question have atleast one correct choice :)))
        if '1' not in all_choices_answers:
            all_choices_answers[best_positive_index] = "1"
        answer = "".join(all_choices_answers)

        return answer
