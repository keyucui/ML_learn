#encoding=utf-8
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import sys


#process the dataset

data = [['青绿', '蜷缩', '浊响','清晰','凹陷','硬滑','是'],
               ['乌黑', '蜷缩', '沉闷','清晰','凹陷','硬滑','是'],
               ['乌黑', '蜷缩', '浊响','清晰','凹陷','硬滑','是'],
               ['青绿', '稍蜷', '浊响','清晰','稍凹','软粘','是'],
               ['乌黑', '稍蜷', '浊响','稍糊','稍凹','软粘','是'],
               ['青绿', '硬挺', '清脆','清晰','平坦','软粘','否'],
               ['浅白', '稍蜷', '沉闷','稍糊','凹陷','硬滑','否'],
               ['乌黑', '稍蜷', '浊响','清晰','稍凹','软粘','否'],
               ['浅白', '蜷缩', '浊响','模糊','平坦','硬滑','否'],
               ['青绿', '蜷缩', '沉闷','稍糊','稍凹','硬滑','否']]

data_test =  [['青绿', '蜷缩', '沉闷','清晰','凹陷','硬滑','是'],
               ['浅白', '蜷缩', '浊响','清晰','凹陷','硬滑','是'],
               ['乌黑', '稍蜷', '浊响','清晰','稍凹','硬滑','是'],
               ['浅白', '硬挺', '清脆','模糊','稍凹','硬滑','否'],
               ['浅白', '蜷缩', '浊响','模糊','平坦','软粘','否'],
               ['青绿', '稍蜷', '浊响','稍糊','凹陷','硬滑','否']]


attributeMap={}
attributeMap['浅白']=0
attributeMap['青绿']=0.5
attributeMap['乌黑']=1
attributeMap['蜷缩']=0
attributeMap['稍蜷']=0.5
attributeMap['硬挺']=1
attributeMap['沉闷']=0
attributeMap['浊响']=0.5
attributeMap['清脆']=1
attributeMap['模糊']=0
attributeMap['稍糊']=0.5
attributeMap['清晰']=1
attributeMap['凹陷']=0
attributeMap['稍凹']=0.5
attributeMap['平坦']=1
attributeMap['硬滑']=0
attributeMap['软粘']=1
attributeMap['否']=0
attributeMap['是']=1

data = np.array(data)
np.random.shuffle(data)
test_targets = np.zeros((7, 1))
train_targets = np.zeros((10, 1))
train_features = np.zeros((10, 6))
test_features = np.zeros((7, 6))
for i in range(len(data)):
    for j in range(len(data[i])):
        if j != 6:
            train_features[i][j] = attributeMap[data[i][j]]
        else:
            train_targets[i][0] = attributeMap[data[i][j]]
for i in range(len(data_test)):
    for j in range(len(data[i])):
        if j != 6:
            test_features[i][j] = attributeMap[data_test[i][j]]
        else:
            test_targets[i][0] = attributeMap[data_test[i][j]]
print('打乱顺序后的train_X')
print(train_features)
print('打乱顺序后的train_Y')
print(train_targets)

# test_features, test_targets = np.array([[4,1],[1, 4]]), np.array([[1],[0]])
# train_features, train_targets = np.array([[2,1],[1,2],[3,2], [3,1],[1,3],[2,3]]), np.array([[1],[0],[1],[1],[0],[0]])
#val_features, val_targets = []


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes ** -0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        self.activation_function = self.sigmoid

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def tangenth(self, x):
        return (1.0 * math.exp(x) - 1.0 * math.exp(-x)) / (1.0 * math.exp(x) + 1.0 * math.exp(-x))

    def softmax(self, inMatrix):
        m, n = np.shape(inMatrix)
        outMatrix = np.mat(np.zeros((m, n)))
        soft_sum = 0
        for idx in range(0, n):
            outMatrix[0, idx] = math.exp(inMatrix[0, idx])
            soft_sum += outMatrix[0, idx]
        for idx in range(0, n):
            outMatrix[0, idx] /= soft_sum
        return outMatrix

    def train(self, inputs_list, targets_list):
        '''
        将输入转置，因为权重矩阵为(hidden_dim * input_dim)，
        需要将inputs改为(input_dim * samples),
        得到(hidden_dim * samples)
        '''
        inputs = np.array(inputs_list, ndmin=2)
        targets = np.array(targets_list, ndmin=1)
        #print('inputs', inputs.shape)
        #print('targets', targets.shape)
        hidden_inputs = np.dot(inputs, self.weights_input_to_hidden)
        #print('hidden_inputs_dim', hidden_inputs.shape)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output,)
        final_outputs = final_inputs
        #print('outputs', final_outputs.shape)
        #print('asd', targets.shape)
        #### Implement the backward pass here ####
        ### Backward pass ###

        output_errors = (targets - final_outputs)
        # errors propagated to the hidden layer
        hidden_errors = np.dot(output_errors, self.weights_hidden_to_output.T)
        # hidden layer gradients
        hidden_grad = hidden_outputs * (1.0 - hidden_outputs)
        #print('hidden_grad', hidden_grad.shape)
        #print('hidden_errors', hidden_errors.shape)                
        #print('input.T', inputs.T.shape)
        #print((hidden_errors * hidden_grad).shape)
        # update hidden-to-output weights
        self.weights_hidden_to_output += self.lr * np.dot(hidden_outputs.T, output_errors)
        # update input-to-hidden weights
        self.weights_input_to_hidden += self.lr * np.dot(inputs.T, (hidden_errors * hidden_grad))
        #2 * 10                                        2*6              6*10     6*10
        
    def predict(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2)

        hidden_inputs = np.dot(inputs, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs  # signals from final output layer

        return final_outputs


def MSE(y, Y):
    return np.mean((y - Y) ** 2)


### Set the hyperparameters here ###
epochs = 40
learning_rate = 0.01
hidden_nodes = 10
output_nodes = 1
batch_size = 56

input_nodes = train_features.shape[1]
network = NeuralNetwork(6, hidden_nodes, output_nodes, learning_rate)
losses = {'train': [], 'validation': []}

for e in range(epochs):

    network.train(train_features, train_targets)

    # Printing out the training progress
    train_loss = MSE(network.predict(train_features), train_targets)
    print("\rProgress: " + str(100 * (e+1) / float(epochs))[:4] \
                     + "% ... Training Mean Square Error : " + str(train_loss)[:5])

    losses['train'].append(train_loss)
    #losses['validation'].append(val_loss)

plt.plot(losses['train'], label='Training loss')
#plt.plot(losses['validation'], label='Validation loss')
plt.legend()
plt.ylim(ymax=1)
plt.xlabel('epochs')
plt.ylabel('Mean Square Error')
plt.show()

predictions = network.predict(test_features)
test_loss = MSE(network.predict(test_features), test_targets)
print('验证集的预测值\n', predictions, '\n')
print('验证集的实际值\n', test_targets,'\n')
print('验证集均方误差为', test_loss)

