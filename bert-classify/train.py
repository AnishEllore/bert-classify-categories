import pandas as pd
import torch
import numpy as np
from core.model import *
import os
curdir = os.getcwd()
PATH = curdir+'/data'
datapath = PATH+'/bbc-text.csv'
print(datapath)
df = pd.read_csv(datapath)
df.head()

df.groupby(['category']).size().plot.bar()


np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df)), int(.9*len(df))])

print(len(df_train),len(df_val), len(df_test))

EPOCHS = 1
model = BertClassifier()
LR = 1e-6
              
#train(model, df_train, df_val, LR, EPOCHS)
evaluate(model, df_test)
infer_data = ["all is well", "VT is good in sports", "Asu is better than usa"]
df_infer = pd.DataFrame(infer_data, columns=['text'])
infer_output = inference(model, df_infer)
infer_output = get_labels(infer_output)
print(infer_output)

torch.save(model.state_dict(), PATH+'/bert-model.pt')

