import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import os

processed_dir = r"E:\project AI\vehicle-tracking-deai\data\processed"
model_file = os.path.join(processed_dir, "lstm_model_multi")

model = load_model(model_file, compile=False)

X = np.load(os.path.join(processed_dir, "X_lstm_multi.npy"))
y = np.load(os.path.join(processed_dir, "y_lstm_multi.npy"))

pred = model.predict(X)

fig = go.Figure()

fig.add_trace(go.Scatter(x=y[:,0], y=y[:,1], mode='lines+markers', name='Real Path'))

fig.add_trace(go.Scatter(x=pred[:,0], y=pred[:,1], mode='lines+markers', name='Predicted Path'))

fig.update_layout(title='Multi-Vehicle Trajectory Prediction',
                  xaxis_title='X', yaxis_title='Y')

fig.show()
