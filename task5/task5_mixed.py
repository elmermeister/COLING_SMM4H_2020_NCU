import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


train_df = pd.read_csv('data/train.csv',encoding='utf-8')
train_df.to_csv('train_df.csv',index=False)


model = ClassificationModel('bert', 'bert-base-chinese', num_labels=4, args={'learning_rate':1e-5,'train_batch_size':256, 'num_train_epochs': 50, 'reprocess_input_data': True, 'overwrite_output_dir': True})
model.train_model(train_df)



test_df = pd.read_csv('data/test.csv',encoding='utf-8')
to_predict = test_df.text.apply(lambda x: x.replace('\n', ' ')).tolist()


preds, outputs = model.predict(to_predict)

sub_df = pd.DataFrame(outputs)


sub_df['text'] = test_df['text']
#sub_df = sub_df["label"]
try:
    sub_df.to_csv('outputs/submission.csv', index=False)
except:
    print('csv are not available')

y_true = list(test_df['label'])
y_pred = preds.tolist()
target_names = ['class 0', 'class 1', 'class 2', 'class 3']
print(classification_report(y_true, y_pred, target_names=target_names))

C=confusion_matrix(y_true, y_pred)
print(C)
with open ('outputs/cm_evaluation.txt','w',encoding='utf-8') as fw:
    fw.write(classification_report(y_true, y_pred, target_names=target_names))
    fw.write('\n')
    fw.write(str(C))
    fw.write('\n')
    fw.write(str((C[0][0]+C[1][1])/(C[0][0]+C[1][1]+C[0][1]+C[1][0])))
    




