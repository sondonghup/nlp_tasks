import uvicorn
from typing import List, Union
import sys
import os
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from argparse import ArgumentParser
from model.spacy_ner import get_result

class Inputs(BaseModel):
    nickName: str = "acer"
    modelName: str = "spacy_ner"
    user_input: str = "please input eng sentence"

description = """
# spacy ner model



"""

tags_metadata = [
    {
        "name" : "spacy_ner",
        "description" : "eng_ner"
    }
]

app = FastAPI(
    title = "spacy ner api",
    description = description
)

@app.post("/acer-lab/ner", tags=["ner"])
def ner(inputs: Inputs):
    try:
        input = inputs.user_input
        target_output = get_result(input)
        error = None

        result = {
            "inputs" : inputs.user_input,
            "ner" : target_output
        }


    except Exception as e:
        result = None
        error = str(e)
    
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