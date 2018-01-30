import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    for i in range((len(series) - window_size)):
        y.append(series[i+window_size])    
        X.append(series[i:(i+window_size)]) 
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5,input_shape=(window_size,1)))
    model.add(Dense(1))
    return model

def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    text = re.sub("[\s+\!\/_,$%^*(+\"\']+|[+——，。\、~@#￥%……&*（）]+", " ",text)
    return text

def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    for i in range(window_size,len(text),step_size):
        outputs.append(text[i])    
        inputs.append(text[(i-window_size):i]) 
    # reshape each 
    inputs = np.asarray(inputs)
    inputs.shape = (np.shape(inputs)[0:2])
    outputs =np.asarray(outputs)
#     outputs.shape = (len(outputs),1)                      
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_dim=58, input_length=window_size, return_sequences=True))
    model.add(LSTM(num_chars, return_sequences=True))
    model.add(LSTM(58))
    return model
