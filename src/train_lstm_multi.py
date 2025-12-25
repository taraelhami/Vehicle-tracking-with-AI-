from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os
checkpoint = ModelCheckpoint("lstm_model_multi_best.keras", save_best_only=True)
processed_dir = r"E:\project AI\vehicle-tracking-deai\data\processed"

X = np.load(os.path.join(processed_dir, "X_lstm_multi.npy"))
y = np.load(os.path.join(processed_dir, "y_lstm_multi.npy"))

# model = Sequential([
#     LSTM(100, activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
#     Dropout(0.2),
#     LSTM(50, activation='relu'),
#     Dense(2)
# ])
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(25, activation='relu'),
    Dense(2)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X, y, epochs=20, batch_size=16, callbacks=[checkpoint])

model.save(os.path.join(processed_dir, "lstm_model_multi"))
print("Model saved successfully!")
