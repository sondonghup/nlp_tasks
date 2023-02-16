from fastapi import FastAPI
import uvicorn
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ElectraForTokenClassification

app = FastAPI()

class TextInput(BaseModel):
    text: str

model_name = "monologg/koelectra-base-finetuned-naver-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ElectraForTokenClassification.from_pretrained(model_name)

@app.post("/analyze_sentiment")
def analyze_sentiment(text_input: TextInput):
    results = []
    inputs = tokenizer(text_input.text, return_tensors="pt")
    outputs = model(**inputs).logits
    result = torch.argmax(outputs.cpu().detach(), axis = -1)[0]
    for token, ner in zip(inputs[0].tokens, result):
        results.append([ner.numpy()])
    return results

if __name__ == "__main__":
    uvicorn.run(app)