import uvicorn
from typing import List, Union
import sys
import os
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from argparse import ArgumentParser
from model.naver_ner import get_model, get_result
from transformers import (
    AutoTokenizer,
    ElectraForTokenClassification
)

class Item(BaseModel):
    nickName: str = "acer"
    modelName: str = "kor_ner"
    user_input: str = "please input kor sentence"

description = """
# naver kor ner model



"""

tags_metadata = [
    {
        "name" : "naver_ner",
        "description" : "kor_ner"
    }
]

origins = [
    "http://localhost",
]

app = FastAPI(
    title = "naver ner api",
    description = description
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/kor_ner")
def kor_ner(inputs: Item):
    
    print(inputs.user_input)
    inputs.user_input=inputs.user_input.replace("\n"," ")
    inputs.user_input=inputs.user_input.replace(u"\xa0",u"")
    input = inputs.user_input

    target_output = tokenizer(input, return_tensors='pt')

    error = None

    result = {
        "inputs" : inputs.user_input,
        "ner" : target_output
    }

    # except Exception as e:
    #     result = None
    #     error = str(e)
    
    return {
        "result" : result,
        "error" : error
    }

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", default = 'config/v0.0.1-ner.json')
    parse = parser.parse_args()
    cfg_path = parse.config
    config = json.load(open(cfg_path, encoding="utf-8"))
    model_name = config['model_name']
    
    tokenizer, model = get_model(model_name) # 허깅페이스에서 받아오는 토크나이저 모델 받아오는 부분

    uvicorn.run(app, host="127.0.0.1", port=8000)