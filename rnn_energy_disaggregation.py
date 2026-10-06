# ==========================================================================
# ComparingBasicRNNArchitectures
# ---------------------------------------------------------------------
# Compares three recurrent architectures -- SimpleRNN, LSTM and GRU -- on
# an energy disaggregation sequence-to-sequence problem: given the total
# (aggregate) household power consumption over a window of time, predict
# the power consumption of a single appliance (a coffee machine) over
# the same window.
#
# Data (provided via eclass, read directly from Google Drive):
#   - CoffeeMachinemaxAgg.txt : scalar, max aggregate consumption seen at
#                               the household's electrical panel (used to
#                               normalize the INPUT values)
#   - CoffeeMachinemaxApp.txt : scalar, max consumption seen at the coffee
#                               machine (used to normalize the OUTPUT values)
#   - Input_Data.txt          : k x m matrix, aggregate power (W), one
#                               sample per minute, over m-minute windows
#   - Output_Data.txt         : k x m matrix, coffee-machine power (W),
#                               same shape/timing as Input_Data.txt
#
# Architecture reference: this script follows the same style as the
# course-provided seq2seqdemo.py (SimpleRNN/LSTM/GRU with
# return_sequences=True, followed by a Dense(1) applied at every
# timestep, Adam optimizer, MeanSquaredError loss).
# ==========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras as ks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
np.random.seed(SEED)
ks.utils.set_random_seed(SEED)
rng = np.random.default_rng(SEED)

# --------------------------------------------------------------------
# 0. Mount Google Drive and load the data
# --------------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

# >>> EDIT THIS PATH to wherever the 4 .txt files actually live in your Drive <<<
DATA_DIR = "/content/drive/MyDrive/Coffee Machine Consumption"

# IMPORTANT: the provided .txt files use a COMMA as the value separator
# (not whitespace), so delimiter=',' is required for np.loadtxt to work.
max_agg = np.loadtxt(f"{DATA_DIR}/CoffeeMachinemaxAgg.txt", delimiter=",")
max_app = np.loadtxt(f"{DATA_DIR}/CoffeeMachinemaxApp.txt", delimiter=",")
input_data = np.loadtxt(f"{DATA_DIR}/Input_Data.txt", delimiter=",")     # shape (k, m)
output_data = np.loadtxt(f"{DATA_DIR}/Output_Data.txt", delimiter=",")  # shape (k, m)

print(f"max_agg = {max_agg}, max_app = {max_app}")
print("Input_Data shape:", input_data.shape)
print("Output_Data shape:", output_data.shape)

k, m = input_data.shape  # k sequences, each m minutes long

# --------------------------------------------------------------------
# IMPORTANT DESIGN NOTE on normalization:
# Inspecting the actual files shows Input_Data.txt / Output_Data.txt
# already contain small-magnitude values (Input roughly in [-1.1, 3.3],
# Output in [-4.0, 2.0], mostly clustered near 0) -- NOT raw Watt-scale
# values comparable to max_agg (1633.7) / max_app (1085.09). Dividing
# them again by max_agg/max_app would shrink everything to a ~0.000x
# range, destroying the training signal.
# We therefore treat Input_Data.txt / Output_Data.txt as ALREADY
# normalized (this is what the max files were used to produce in the
# first place), and use max_agg / max_app ONLY to convert back to real
# Watts at the end, wherever the assignment explicitly asks for
# denormalized values (plots in Watts, and the RMSE/MAE/MaxError metrics).
# If you find out from your professor that the files are meant to be
# raw (unnormalized) instead, simply set RAW_DATA = True below.
# --------------------------------------------------------------------
RAW_DATA = False

if RAW_DATA:
    input_norm = input_data / max_agg
    output_norm = output_data / max_app
else:
    input_norm = input_data
    output_norm = output_data

# --------------------------------------------------------------------
# 1. Plot 3 random raw sequences (aggregate + appliance), dual y-axis
#    At least one of them is guaranteed to show real device activity.
# --------------------------------------------------------------------
ACTIVITY_THRESHOLD_NORM = 0.05 * max_app if RAW_DATA else 0.01
device_active_rows = np.where(output_data.max(axis=1) > ACTIVITY_THRESHOLD_NORM)[0]

plot_rows = list(rng.choice(k, size=2, replace=False))
if len(device_active_rows) > 0:
    plot_rows.append(int(rng.choice(device_active_rows)))
else:
    plot_rows.append(int(rng.choice(k)))

for row in plot_rows:
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()  # secondary y-axis for the appliance signal

    line1, = ax1.plot(input_data[row], color="tab:blue", label="Συνολική κατανάλωση (aggregate)")
    line2, = ax2.plot(output_data[row], color="tab:orange", label="Κατανάλωση καφετιέρας (appliance)")

    ax1.set_xlabel("Λεπτά")
    ax1.set_ylabel("Συνολική κατανάλωση (W)", color="tab:blue")
    ax2.set_ylabel("Κατανάλωση καφετιέρας (W)", color="tab:orange")
    ax1.set_title(f"Ενδεικτική ακολουθία #{row}")
    ax1.legend(handles=[line1, line2], loc="upper right")

    plt.tight_layout()
    plt.savefig(f"sample_sequence_{row}.png", dpi=150, bbox_inches="tight")
    plt.show()

# --------------------------------------------------------------------
# 2. Reshape to (num_sequences, timesteps, num_features=1) and build
#    train / validation / test splits
# --------------------------------------------------------------------
X = input_norm.reshape(k, m, 1).astype("float32")
y = output_norm.reshape(k, m, 1).astype("float32")

indices = np.arange(k)
rng.shuffle(indices)

# 70% train / 10% validation / 20% test. The validation set is carved out
# of the training data and is independent of both train and test.
train_end = int(0.7 * k)
val_end = int(0.8 * k)
train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Train sequences: {len(train_idx)} | Validation: {len(val_idx)} | Test: {len(test_idx)}")

# --------------------------------------------------------------------
# 3. Model builders (RNN / LSTM / GRU)
# --------------------------------------------------------------------
def build_model(cell_type):
    """Builds a single-layer recurrent seq2seq regressor.
    cell_type: one of 'RNN', 'LSTM', 'GRU'."""
    layer_cls = {"RNN": SimpleRNN, "LSTM": LSTM, "GRU": GRU}[cell_type]
    model = Sequential(name=cell_type)
    # return_sequences=True -> one hidden output per timestep (seq2seq)
    model.add(layer_cls(64, input_shape=(m, 1), return_sequences=True))
    # Dense(1) is applied independently at every timestep, producing the
    # predicted appliance power for that minute.
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss=MeanSquaredError())
    return model


EPOCHS = 100      # upper bound; EarlyStopping will typically stop earlier.
                  # Satisfies the ">= 30 epochs" requirement as a ceiling.
BATCH_SIZE = 32

models = {}
histories = {}

for cell_type in ["RNN", "LSTM", "GRU"]:
    print(f"\n=== Training {cell_type} ===")
    model = build_model(cell_type)

    # Stop training if validation loss doesn't improve for 5 consecutive epochs
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # verbose=1 prints training AND validation loss every epoch
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )
    models[cell_type] = model
    histories[cell_type] = history

    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title(f"{cell_type} - MSE loss (training vs validation)")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss (normalized scale)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"loss_{cell_type}.png", dpi=150, bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------
# 4. Per-sequence RMSE / MAE / Max Error on DENORMALIZED data
# --------------------------------------------------------------------
def per_sequence_errors(model, X_set, y_set):
    """Returns 3 arrays (one value per sequence): RMSE, MAE, max error,
    computed on denormalized (real Watt) values, as required."""
    y_pred_norm = model.predict(X_set, verbose=0)
    y_pred = y_pred_norm.squeeze(-1) * max_app   # denormalize predictions
    y_true = y_set.squeeze(-1) * max_app         # denormalize ground truth

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))
    mae = np.mean(np.abs(y_true - y_pred), axis=1)
    max_err = np.max(np.abs(y_true - y_pred), axis=1)
    return rmse, mae, max_err


summary_rows = []
test_errors = {}  # keep per-sequence test errors for the required plots

for cell_type, model in models.items():
    for set_name, X_set, y_set in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        rmse, mae, max_err = per_sequence_errors(model, X_set, y_set)
        summary_rows.append({
            "Technique": cell_type,
            "Set": set_name,
            "RMSE_mean": rmse.mean(),
            "MAE_mean": mae.mean(),
            "MaxError_mean": max_err.mean(),
        })
        if set_name == "Test":
            test_errors[cell_type] = {"rmse": rmse, "mae": mae, "max_err": max_err}

results_df = pd.DataFrame(summary_rows)
print("\n=== Summary of regression errors (denormalized, Watt) ===")
print(results_df)
results_df.to_csv("erotima1_results.csv", index=False)
print("Saved erotima1_results.csv")

# One plot per error type, comparing all 3 techniques across test sequences
error_labels = {"rmse": "RMSE (W)", "mae": "MAE (W)", "max_err": "Max Error (W)"}
for err_key, err_label in error_labels.items():
    plt.figure(figsize=(10, 4))
    for cell_type in ["RNN", "LSTM", "GRU"]:
        plt.plot(test_errors[cell_type][err_key], marker="o", markersize=3, label=cell_type)
    plt.title(f"{err_label} ανά test sequence (3 τεχνικές)")
    plt.xlabel("Test sequence index")
    plt.ylabel(err_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{err_key}_per_test_sequence.png", dpi=150, bbox_inches="tight")
    plt.show()

# --------------------------------------------------------------------
# 5. Select 4 test rows (2 device-ON, 2 device-OFF) and plot predictions
#    of all 3 models against the real consumption
# --------------------------------------------------------------------
test_output_denorm = y_test.squeeze(-1) * max_app
on_mask = test_output_denorm.max(axis=1) > 0.05 * max_app
off_mask = ~on_mask

on_positions = np.where(on_mask)[0]
off_positions = np.where(off_mask)[0]

n_on = min(2, len(on_positions))
n_off = min(2, len(off_positions))
chosen_on = list(rng.choice(on_positions, size=n_on, replace=False)) if n_on > 0 else []
chosen_off = list(rng.choice(off_positions, size=n_off, replace=False)) if n_off > 0 else []
chosen = chosen_on + chosen_off

for pos in chosen:
    status = "ON (η συσκευή λειτουργούσε)" if pos in on_positions else "OFF (η συσκευή ήταν σβηστή)"
    plt.figure(figsize=(10, 4))
    plt.plot(test_output_denorm[pos], color="black", linewidth=2.2, label="Πραγματική κατανάλωση")
    for cell_type, model in models.items():
        pred = model.predict(X_test[pos:pos + 1], verbose=0).squeeze() * max_app
        plt.plot(pred, linestyle="--", label=f"{cell_type} πρόβλεψη")
    plt.title(f"Test sequence #{pos} — {status}")
    plt.xlabel("Λεπτά")
    plt.ylabel("Κατανάλωση καφετιέρας (W)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"prediction_test_seq_{pos}.png", dpi=150, bbox_inches="tight")
    plt.show()

print("\nΟλοκληρώθηκε. Αρχεία που δημιουργήθηκαν:")
print(" - sample_sequence_*.png (3 αρχεία)")
print(" - loss_RNN.png, loss_LSTM.png, loss_GRU.png")
print(" - erotima1_results.csv")
print(" - rmse_per_test_sequence.png, mae_per_test_sequence.png, max_err_per_test_sequence.png")
print(" - prediction_test_seq_*.png (4 αρχεία)")
