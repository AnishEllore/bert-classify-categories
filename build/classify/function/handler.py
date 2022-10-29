import json
import os
from pathlib import PurePath
from typing import Any, List
import time

import torch
from torch.autograd import Variable

from .core import const, model, utils

FUNCTION_ROOT = os.environ.get("function_root", "/root/function/")

time_model_load_start = time.monotonic()
# init model
RNN = model.RNN(const.N_LETTERS, const.N_HIDDEN, const.N_CATEGORIES)
# fill in weights
RNN.load_state_dict(
    torch.load(str(PurePath(FUNCTION_ROOT, "data/char-rnn-classification.pt")))
)
time_model_load_end = time.monotonic()

def predict(line: str, n_predictions: int = 3) -> List[Any]:
    output = model.evaluate(Variable(utils.line_to_tensor(line)), RNN)

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions: List[Any] = []

    for i in range(n_predictions):
        value = str(topv[0][i]).split("tensor")[1]
        category_index = topi[0][i]
        predictions += [(value, const.ALL_CATEGORIES[category_index])]

    return predictions

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
