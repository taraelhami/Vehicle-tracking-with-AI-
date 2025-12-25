import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

processed_dir = r"E:\project AI\vehicle-tracking-deai\data\processed"

# prediction
X = np.load(os.path.join(processed_dir, "X_lstm_multi.npy"))
y = np.load(os.path.join(processed_dir, "y_lstm_multi.npy"))

from tensorflow.keras.models import load_model
model_file = os.path.join(processed_dir, "lstm_model_multi")
model = load_model(model_file, compile=False)

pred = model.predict(X)

#  RMSE and MAE
rmse = np.sqrt(mean_squared_error(y, pred))
mae = mean_absolute_error(y, pred)

print(f"Model Evaluation:\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}")
