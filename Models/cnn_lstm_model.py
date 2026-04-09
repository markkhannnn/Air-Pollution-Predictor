from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout

def build_cnn_lstm(input_shape, output_dim):
    model = Sequential()

    model.add(Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu',
        input_shape=input_shape
    ))
    model.add(Dropout(0.2))

    model.add(LSTM(64))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_dim))  # multi-output

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model