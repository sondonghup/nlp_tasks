import torch
from transformers import (
    AutoTokenizer,
    ElectraForTokenClassification,
    BertForSequenceClassification
)

def get_model(NAME):
    tokenizer = AutoTokenizer.from_pretrained(NAME)
    model = BertForSequenceClassification.from_pretrained(NAME)
    model = model.eval()

    return tokenizer, model

def get_result(inputs, tokenizer, model):
    result = []
    tokenized_sent = make_input(inputs, tokenizer)
    ner_sent = model(**tokenized_sent).logits
    ner_sent = torch.argmax(ner_sent.cpu().detach(), axis = -1)[0]
    for token, ner in zip(tokenized_sent[0].tokens[1:-1], ner_sent[1:-1]):
        result.append([token, ner.numpy()])

    return result

def make_input(inputs, tokenizer):
    return tokenizer(inputs, truncation = True, return_tensors="pt")