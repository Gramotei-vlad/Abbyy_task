from typing import List, Dict
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from seqeval.metrics import f1_score


def get_metrics(labels: List[List[str]], predictions: List[List[str]]) -> Dict[str, float]:
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    dict_metrics = {
        'precision': precision,
        'recall': recall,
        'f1 score': f1,
    }

    return dict_metrics
