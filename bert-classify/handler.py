import json
import os
from pathlib import PurePath
from typing import Any, List
import time
import pandas as pd
import torch
from torch.autograd import Variable

from .core.model import *

FUNCTION_ROOT = os.environ.get("function_root", "/root/function/")

time_model_load_start = time.monotonic()
# init model
model = BertClassifier()
# fill in weights
model.load_state_dict(
    torch.load(str(PurePath(FUNCTION_ROOT, "data/bert-model.pt")))
)
time_model_load_end = time.monotonic()

def predict(line: str, n_predictions: int = 3) -> List[Any]:
    infer_data = []
    infer_data.append(line)
    df_infer = pd.DataFrame(infer_data, columns=['text'])
    infer_output = inference(model, df_infer)
    infer_output = get_labels(infer_output)
    return infer_output

def convert_to_ms(num):
    return int(round(num*1000))

def handle(req: bytes) -> str:
    """handle a request to the function
    Args:
        req (bytes): request body
    """

    if not req:
        return json.dumps({"error": "No input provided", "code": 400})
    time_point_1 = time.monotonic()
    name = str(req)
    time_point_2 = time.monotonic()
    output = predict(name)
    time_point_3 = time.monotonic()
    #output = os.environ.get('MODEL_NAME', output)
    #print(output)
    model_load_time = convert_to_ms(time_model_load_end - time_model_load_start)
    prediction_time = convert_to_ms(time_point_3 - time_point_2)
    function_prediction_time = convert_to_ms(time_point_3 - time_point_1)
    output = str(output)+"\nmodel_load_time:{}ms, prediction_time:{}ms, function_exec_time:{}ms\n".format(model_load_time, prediction_time, function_prediction_time)
    return json.dumps(output)
