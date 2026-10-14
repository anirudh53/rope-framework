"""
Regenerates f10_tiny.f10bin + f10_tiny_expected.json (golden data for test_f10_ensemble.cpp).

Two small members with fixed random weights -- an LSTM (lookback 7, sigmoid/relu/tanh layers)
and an MLP (lookback 10) -- written in the .f10bin layout, plus the forecast the reference
implementation (Keras predict + the forecaster's dynamic_pred / dynamic_pred_lstm loops)
produces for a fixed history. Needs TensorFlow/Keras:

    python make_f10_fixture.py
"""
import json
import os
import struct

import numpy as np
from tensorflow import keras

HERE = os.path.dirname(os.path.abspath(__file__))
ACT = {"linear": 0, "tanh": 1, "sigmoid": 2, "relu": 3}
MEAN, STD, MAX_LB, AVG = 129.39226979, 54.35683083, 22, 12
rng = np.random.default_rng(7)


def randomize(m):
    m.set_weights([rng.normal(0, 0.4, w.shape).astype(np.float32) for w in m.get_weights()])
    return m


lstm = randomize(keras.Sequential([
    keras.Input((14, 1)),
    keras.layers.LSTM(5, activation="sigmoid", recurrent_activation="sigmoid", return_sequences=True),
    keras.layers.LSTM(3, activation="relu", recurrent_activation="sigmoid"),
    keras.layers.Dense(4, activation="tanh"),
    keras.layers.Dense(1, activation="linear"),
]))
mlp = randomize(keras.Sequential([
    keras.Input((20,)),
    keras.layers.Dense(6, activation="relu"),
    keras.layers.Dense(3, activation="sigmoid"),
    keras.layers.Dense(1, activation="linear"),
]))
members = [(7, lstm, True), (10, mlp, False)]


def dynamic_pred(model, x_row, n):            # reference: f10_forecaster.dynamic_pred
    yhat, xhat = [], x_row.reshape(1, -1).copy()
    for _ in range(n):
        back_ave = np.mean(np.diff(xhat[0, ::-2])) + xhat[0, -1]
        pred = model.predict(xhat, batch_size=1, verbose=0)[0]
        yhat.append(pred[0])
        xhat = np.roll(xhat, shift=-2, axis=-1)
        xhat[0, -2:] = np.array([pred[0], back_ave])
    return np.array(yhat)


def dynamic_pred_lstm(model, x_row, n):       # reference: f10_forecaster.dynamic_pred_lstm
    yhat, xhat = [], x_row.reshape(1, -1, 1).copy()
    for _ in range(n):
        back_ave = np.mean(np.diff(xhat[0, ::-2, 0])) + xhat[0, -1, 0]
        pred = model.predict(xhat, batch_size=1, verbose=0)[0]
        yhat.append(float(pred[0]))
        xhat = np.roll(xhat, shift=-2, axis=1)
        xhat[0, -2:, 0] = np.array([float(pred[0]), back_ave])
    return np.array(yhat)


# history: 40 days of a smooth series with a 27-day rotation signal
hist = 150 + 30 * np.sin(2 * np.pi * np.arange(40) / 27.0) + 5 * np.cos(np.arange(40))
ba = np.array([hist[max(0, d - AVG + 1):d + 1].mean() for d in range(len(hist))])
flat = np.column_stack([hist[-MAX_LB:], ba[-MAX_LB:]]).reshape(-1)
x = (flat - MEAN) / STD
n_days = 5
preds = []
for lb, m, seq in members:
    xl = x[-2 * lb:]
    preds.append((dynamic_pred_lstm if seq else dynamic_pred)(m, xl, n_days) * STD + MEAN)
preds = np.array(preds)

with open(os.path.join(HERE, "f10_tiny.f10bin"), "wb") as f:
    f.write(struct.pack("<II", 0x52504631, 1))
    f.write(struct.pack("<dd", MEAN, STD))
    f.write(struct.pack("<IIII", MAX_LB, 2, AVG, len(members)))
    for lb, m, seq in members:
        f.write(struct.pack("<III", lb, 0 if seq else 1, len(m.layers)))
        for layer in m.layers:
            w = layer.get_weights()
            if isinstance(layer, keras.layers.LSTM):
                f.write(struct.pack("<IIIIII", 0, ACT[layer.activation.__name__],
                                    ACT[layer.recurrent_activation.__name__], int(layer.return_sequences),
                                    w[0].shape[0], w[1].shape[0]))
            else:
                f.write(struct.pack("<IIIIII", 1, ACT[layer.activation.__name__], 0, 0,
                                    w[0].shape[0], w[0].shape[1]))
            for a in w:
                f.write(np.asarray(a, "<f4").reshape(-1).tobytes())

json.dump({"history": hist.tolist(), "n_days": n_days,
           "members": preds.tolist(),
           "mean": preds.mean(axis=0).tolist(), "std": preds.std(axis=0).tolist()},
          open(os.path.join(HERE, "f10_tiny_expected.json"), "w"), indent=1)
print("mean", preds.mean(axis=0), "std", preds.std(axis=0))
