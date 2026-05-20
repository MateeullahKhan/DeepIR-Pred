# models.py
import keras
from keras.layers import Dense, Dropout, Input, Bidirectional, GRU, LSTM, Conv1D, Flatten, BatchNormalization, MaxPooling1D

def get_dnn_model(input_shape):
    model = keras.Sequential([
        Input(shape=input_shape),
        Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
        ])
    return model

def get_cnn1d_model(input_shape):
    model = keras.Sequential([
        Input(shape=input_shape),
        Conv1D(256, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

def get_bilstm_model(input_shape):
    model = keras.Sequential([
        Input(shape=input_shape),
        Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
        LayerNormalization(),
        Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
        LayerNormalization(),
        Bidirectional(keras.layers.LSTM(64, dropout=0.3, recurrent_dropout=0.2)),
        LayerNormalization(),
        Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(1e-4))
        ])
    return model

def get_gru_model(input_shape):
    model = keras.Sequential([
        Input(shape=input_shape),
        GRU(256, return_sequences=True),
        LayerNormalization(),
        Dropout(0.4),
        GRU(128, return_sequences=False),
        LayerNormalization(),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation='relu'),
        LayerNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
        ])
    return model

def get_bigru_model(input_shape):
    model = keras.Sequential([
        Input(shape=input_shape),
        Bidirectional(keras.layers.GRU(256, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)),
        LayerNormalization(),
        Bidirectional(keras.layers.GRU(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)),
        LayerNormalization(),
        Dense(64, activation='relu'),
        LayerNormalization(epsilon=1e-6),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
        ])
    return model