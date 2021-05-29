import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

train_x = []

# Store csv file in a Pandas DataFrame
df = pd.read_csv('/home/lijiali1025/projects/2D_motion/2D_motion/data/train_15_v5.csv')
print(df.shape)

# Scaling the input data
sc = MinMaxScaler()
label_sc = MinMaxScaler()
data = sc.fit_transform(df.values)

# Obtaining the Scale for the labels(usage data) so that output can be re-scaled to actual value during evaluation
label_sc.fit(df.iloc[:, 7:22].values)

# Store csv file in a Pandas DataFrame
df2 = pd.read_csv('/home/lijiali1025/projects/2D_motion/2D_motion/data/hunhe2_15.csv')

# Scaling the input data
sc2 = MinMaxScaler()
label_sc2 = MinMaxScaler()
data2 = sc2.fit_transform(df2.values)

# Obtaining the Scale for the labels(usage data) so that output can be re-scaled to actual value during evaluation
label_sc2.fit(df2.iloc[:, 7:22].values)

# Define lookback period and split inputs/labels
lookback = 25
inputs = np.zeros((len(data) - lookback, lookback, 7))
labels = np.zeros((len(data) - lookback, 15))
print(inputs.shape)
print(labels.shape)
for i in range(lookback, len(data)):
    to_add = np.zeros((lookback, 7))
    tm = data[i - lookback:i]
    for t in range(lookback):
        to_add[t] = tm[t][0:7]
    inputs[i - lookback] = to_add
    labels[i - lookback] = data[i][7:22]
inputs = inputs.reshape(-1, lookback, 7)
labels = labels.reshape(-1, 15)
print(inputs.shape)
print(labels.shape)

# Define lookback period and split inputs/labels
lookback2 = 25
inputs2 = np.zeros((len(data2) - lookback2, lookback2, 7))
labels2 = np.zeros((len(data2) - lookback2, 15))
print(inputs2.shape)
print(labels2.shape)
for i in range(lookback2, len(data2)):
    to_add = np.zeros((lookback2, 7))
    tm = data2[i - lookback2:i]
    for t in range(lookback2):
        to_add[t] = tm[t][0:7]
    inputs2[i - lookback2] = to_add
    labels2[i - lookback2] = data2[i][7:22]
inputs2 = inputs2.reshape(-1, lookback2, 7)
labels2 = labels2.reshape(-1, 15)
print(inputs2[0])
# print(labels2)

labss = torch.from_numpy(np.array(labels2))
# print(labss)
ss = label_sc2.inverse_transform(labss.numpy()).reshape(-1)
# print(labss.numpy().reshape(-1))
# print(ss)

# Split data into train/test portions and combining all data from different files into a single array
test_portion = int(0.1 * len(inputs))
train_x = inputs[:-test_portion]
train_y = labels[:-test_portion]
test_x = inputs[-test_portion:]
test_y = labels[-test_portion:]
# print(test_y)
labs = torch.from_numpy(np.array(test_y))
label_sc.inverse_transform(labs)

test_x2 = inputs2[:]
test_y2 = labels2[:]
labs2 = torch.from_numpy(np.array(test_y2))
label_sc2.inverse_transform(labs2)
print(test_x2[0])

batch_size = 100

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)
input_dim = next(iter(train_loader))[0].shape
print(input_dim)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        '''
        The out is with shape (batch, seq_len, num_directions * hidden_size)
        use out[:, -1] means get the hidden information of the final seq
        so it is the hidden information at the end of the for loop
        which means it is a N to 1 structure 
        '''
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


def train(train_loader, learn_rate, hidden_dim=256, EPOCHS=80, model_type="GRU"):
    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 15
    n_layers = 3
    # Instantiating the model
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    # Start training loop
    for epoch in range(1, EPOCHS + 1):
        start_time = time.clock()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            h = h.data
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            # if counter%20 == 0:
            print(
                "Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader),
                                                                                     avg_loss / counter))
        current_time = time.clock()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
        print("Time Elapsed for Epoch: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model


def evaluate(model, test_x, test_y, label_sc):
    model.eval()
    outputs = []
    targets = []
    start_time = time.clock()

    inp = torch.from_numpy(np.array(test_x))
    labs = torch.from_numpy(np.array(test_y))
    h = model.init_hidden(inp.shape[0])
    out, h = model(inp.to(device).float(), h)
    outputs.append(label_sc.inverse_transform(out.cpu().detach().numpy()).reshape(-1))
    targets.append(label_sc.inverse_transform(labs.numpy()).reshape(-1))

    print("Evaluation Time: {}".format(str(time.clock() - start_time)))
    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i] - targets[i]) / (targets[i] + outputs[i]) / 2) / len(outputs)
    print("sMAPE: {}%".format(sMAPE * 100))
    return outputs, targets, sMAPE


# lr = 0.001
# gru_model = train(train_loader, lr, model_type="GRU")
#
# torch.save(gru_model.state_dict(), "entire_model_sensor_no_shuffle_16.pt")
gru_model = GRUNet(7, 256, 15, 3)
gru_model.load_state_dict(torch.load('entire_model_sensor_best.pt'))
gru_model.cuda()
gru_outputs, targets, gru_sMAPE = evaluate(gru_model, test_x2, test_y2, label_sc2)

gru_numpy = gru_outputs[0].reshape(-1, 15)
empty_numpy = np.zeros([1804-lookback, 30])
for i in range(0, 15):
    empty_numpy[:,2*(i+1)-1] = gru_numpy[:,i]

y_only_numpy = empty_numpy
x_df = pd.read_csv('/home/lijiali1025/projects/2D_motion/2D_motion/data/only_x_1.csv')
x_numpy = x_df.to_numpy()
x_numpy = x_numpy[0:(1804-lookback) ,:]
for i in range(0,15):
    y_only_numpy[:,2*i] = x_numpy[:,i]

final_numpy = y_only_numpy
np.savetxt("predict_sensor_only_no.csv", final_numpy, delimiter=",")


# #
# # print('gg')
