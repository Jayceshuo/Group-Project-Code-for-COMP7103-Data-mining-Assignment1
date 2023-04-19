import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.utils.data.dataset as Dataset
import torch.utils.data as data
from torch import nn, optim
import torch.nn.functional as F
from sklearn import metrics
from sklearn.model_selection import train_test_split



class Net(nn.Module):
    def __init__(self,n_input,n_hidden1,n_hidden2,n_hidden3, n_hidden4, n_hidden5,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1,n_hidden2)
        self.hidden3 = nn.Linear(n_hidden2,n_hidden3)
        self.hidden4 = nn.Linear(n_hidden3,n_hidden4)
        self.hidden5 = nn.Linear(n_hidden4,n_hidden5)
        self.predict = nn.Linear(n_hidden5,n_output)
        
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self,input):
        out = self.hidden1(input)
        out = F.elu(out)
        out = self.hidden2(out)
        out = self.dropout(F.elu(out))
        out = self.hidden3(out)
        out = self.dropout(F.elu(out))
        out = self.hidden4(out)
        out = F.elu(out)
        out = self.hidden5(out)
        out = F.elu(out)
        out =self.predict(out) 
        out = out.squeeze(-1)

        return out
    
    
#parameters setting
batch_size = 128
epochs = 1500
learning_rate = 0.01
input_layer = 6
hidden_layer1 = 64
hidden_layer2 = 128
hidden_layer3 = 128
hidden_layer4 = 64
hidden_layer5 = 32
output_layer = 1

#data loader and preprocessing
Data = pd.read_csv('./rawdata_train.csv')
Data_X = Data.drop(['build-Imp'], axis = 1)
Data_Y = Data['build-Imp']
Data_X = (Data_X - Data_X.mean())/Data_X.std()
X_train, X_val, Y_train, Y_val = train_test_split(Data_X, Data_Y,test_size=0.2)
Data_test = pd.read_table('./testdata.txt',sep='\t')
X_test = Data_test.drop(['build-Imp'], axis = 1)
Y_test = Data_test['build-Imp']
X_test = (X_test - X_test.mean())/X_test.std()

train_set = TensorDataset(torch.from_numpy(X_train.values).to(torch.float32),torch.from_numpy(Y_train.values).to(torch.float32))
val_set = TensorDataset(torch.from_numpy(X_val.values).to(torch.float32),torch.from_numpy(Y_val.values).to(torch.float32))



train_loader = DataLoader(train_set,batch_size=batch_size, shuffle=True,drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size,drop_last=False)

#model construction
Regression_model = Net(input_layer,
                       hidden_layer1,
                       hidden_layer2,
                       hidden_layer3,
                       hidden_layer4,
                       hidden_layer5,
                       output_layer)
optimizer = torch.optim.AdamW(Regression_model.parameters(), weight_decay=0.01, lr=learning_rate)
#optimizer = torch.optim.SGD(Regression_model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adam(Regression_model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()
#criterion = torch.nn.SmoothL1Loss()

#training process
train_loss_list, val_loss_list = [], []
Best_val_loss = np.inf
print('Start training:')
for i in range(epochs):
    tot_train_loss = 0
    for features, results in train_loader:
        optimizer.zero_grad()
        
        predict = Regression_model(features)
        loss = criterion(predict, results)
        tot_train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    tot_val_loss = 0
        
    with torch.no_grad():
        for features, results in val_loader:
            predict = Regression_model(features)
            loss = criterion(predict, results)
            tot_val_loss += loss.item()

    train_loss = tot_train_loss / len(train_loader.dataset)
    val_loss = tot_val_loss / len(val_loader.dataset)

    # At completion of epoch
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    
        
    print("Epoch: {}/{}.. ".format(i+1, epochs),
          "Training Loss: {:.3f}.. ".format(train_loss),
          "Val Loss: {:.3f}.. ".format(val_loss),
          )

print('Model trained.')

#loss visulization
plt.plot(train_loss_list, label='Training loss')
plt.plot(val_loss_list, label='Validation loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('Training loss and Validation loss')
plt.legend(frameon=False)
plt.savefig('Loss Curve')


#result visulization
plt.clf()
predict_test = Regression_model(torch.from_numpy(X_test.values).to(torch.float32))
Y_test = Y_test.reset_index(drop = True)
plt.plot(predict_test.detach().numpy(), label='prediction value',c = '#0343df')
plt.plot(Y_test, label='real value', c = '#f97306')
#plt.xlabel('')
plt.ylabel('build-Imp')
plt.title('prediction and real value')
plt.legend(frameon=False)
plt.savefig('Result')

#evaluation process
print('...Starting evalutaion...')
print('Evaluation results:')
mse = metrics.mean_absolute_percentage_error(predict_test.detach().numpy(),Y_test)
print('Mean Absolute Percentage Error:',mse)
R_square = metrics.r2_score(predict_test.detach().numpy(),Y_test)
print('R_square:',R_square)