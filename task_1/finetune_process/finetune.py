import pandas as pd
from simpletransformers.classification import MultiLabelClassificationModel
from sklearn.model_selection import train_test_split
from transformers import ElectraModel, ElectraTokenizer
import torch
from transformers import XLNetTokenizer,XLNetModel
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

df = pd.read_csv('data/train.csv',encoding='utf-8')
df['labels'] = list(zip(
df.a.tolist()
))

print('start')
list_index = []
for each in df["comment_text"]:
    list_index.append(str(each))

print('end')
df["comment_text"] = list_index
df['text'] = df['comment_text']

tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_large")

train_df, eval_df = train_test_split(df, test_size=0.2)

model = MultiLabelClassificationModel('xlnet','clue/xlnet_chinese_large' ,num_labels=26, args={'reprocess_input_data': True,'overwrite_output_dir': True,'train_batch_size':8, 'gradient_accumulation_steps':16, 'learning_rate': 1e-5, 'num_train_epochs':6, 'max_seq_length': 512})

# Train the model
model.train_model(train_df)





test_df = pd.read_csv('data/test.csv',encoding='utf-8')

test_df["comment_text"] = test_df["comment_text"].astype(str)
to_predict = test_df.comment_text.apply(lambda x: x.replace('\n', ' ')).tolist()



preds, outputs = model.predict(to_predict)


sub_df = pd.DataFrame(outputs, columns=['a'])



try:
    sub_df.to_csv('outputs/submission.csv', index=False)
except:
    print('csv are not available')
















