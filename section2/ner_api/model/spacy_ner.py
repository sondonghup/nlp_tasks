import spacy

nlp = spacy.load("en_core_web_sm")

def get_result(inputs):
    result = []
    doc = nlp(inputs)
    for entity in doc.ents:
        result.append([entity.text, entity.label_])

    return result