import uvicorn
import os
from typing import List, Union
import sys
import os

from fastapi import FastAPI
from pydantic import BaseModel

from logger import get_logger
from argparse import ArgumentParser
from model.nltk_ner import get_result

class Inputs(BaseModel):
    nickName: str = "acer"
    modelName: str = "nltk_ner"
    user_input: str = "please input eng sentence"

description = """
# nltk ner model



"""

tags_metadata = [
    {
        "name" : "nltk_ner",
        "description" : "eng_ner"
    }
]

app = FastAPI(
    title = "nltk ner api",
    description = description
)

@app.post("/acer-lab", tags=["ner"])
def ner(inputs: Inputs):
    try:
        print(inputs.user_input)

        logger.info(f"User Inputs : {inputs.user_input}")
        input = inputs.user_input
        target_output = get_result(input)
        ner_output = 

    except:
        pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default = 'config/v0.0.1-ner.json')
    parse = parser.parse_args()
    target_tokenizer, target_models
/bespin-global/klue-roberta-base-ner/blob/main/vocab.txt