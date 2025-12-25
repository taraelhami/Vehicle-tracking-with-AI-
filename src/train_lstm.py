import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

processed_dir = r"E:\project AI\vehicle-tracking-deai\data\processed"

X = np.load(os.path.join(processed_dir, "X_lstm.npy"))
y = np.load(os.path.join(processed_dir, "y_lstm.npy"))

model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    Dense(2) 
])

model.compile(optimizer='adam', loss='mse')
model.summary()

#train model
model.fit(X, y, epochs=50, batch_size=8, verbose=1)

# save model
model_file = os.path.join(processed_dir, "lstm_model.h5")
model.save(model_file)
print(f"LSTM model saved at {model_file}")
