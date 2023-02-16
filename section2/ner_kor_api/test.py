import uvicorn
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import(
    AutoTokenizer,
    ElectraForTokenClassification
)

huggingface_name = 'monologg/koelectra-base-finetuned-naver-ner'

app = FastAPI()

class Item(BaseModel):
    text : str = "진짜 안되면 나한테 쳐맞는다"

tokenizer = AutoTokenizer.from_pretrained(huggingface_name)
model = ElectraForTokenClassification.from_pretrained(huggingface_name)

@app.post("/kor_ner")
def kor_ner(item: Item):
    tokenized_input = tokenizer(item.text, return_tensors='pt')
    result = model(**tokenized_input).logits
    predict = torch.argmax(result.cpu().detach(), axis = -1)[0]
    output = { "result" : tokenized_input}
    return output

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)