import pandas as pd
import os
from sklearn.model_selection import KFold
#加载数据集
X = os.listdir('./tpu-getting-started/Image')
# Y = '训练集对应标签'
# X = []
id = []
label = []
for i in range(len(X)):
    file_name = X[i].split('.')[0]
    x = file_name.split("_")[0]
    y = file_name.split("_")[1]
    id.append(x)
    label.append(y)

df = pd.DataFrame(data={'x':X,'id':id,'label':label})
kf = KFold(n_splits=5)

for i, (train_index, val_index) in enumerate(kf.split(X)):
    train_data = df.iloc[train_index]
    val_data = df.iloc[val_index]
    train_data.to_csv('./tpu-getting-started/csv/'+'train_'+str(i)+'.csv')
    val_data.to_csv('./tpu-getting-started/csv/'+'val_'+str(i)+'.csv')




