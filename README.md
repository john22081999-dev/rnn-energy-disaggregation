# RNN Energy Disaggregation — SimpleRNN vs LSTM vs GRU

A sequence-to-sequence deep-learning project that compares three recurrent
architectures on an **energy disaggregation** problem: given a household's total
(aggregate) power consumption over a time window, predict the consumption of a
single appliance (a coffee machine) over the same window.

## Problem

- **Input:** aggregate household power (W), one sample per minute, in fixed-length
  windows.
- **Output:** the coffee machine's power (W) over the same window (same shape).
- A seq2seq setup: the model reads the whole input window and predicts the whole
  appliance window (a `Dense(1)` applied at every timestep).

## Approach

- Compares **SimpleRNN**, **LSTM** and **GRU** layers (with
  `return_sequences=True`), trained with the Adam optimizer and MSE loss.
- Inputs and outputs are normalized using the dataset's max-consumption scalars,
  then denormalized for interpretable error metrics.

## Tech stack

- Python 3
- TensorFlow / Keras
- numpy, pandas, matplotlib

## Dataset

The data was **provided by the university course** (via eclass) and is **not
redistributed here**. The script expects four text files (comma-separated):

```
CoffeeMachinemaxAgg.txt   # scalar: max aggregate consumption (input normalizer)
CoffeeMachinemaxApp.txt   # scalar: max appliance consumption (output normalizer)
Input_Data.txt            # k x m matrix: aggregate power per minute
Output_Data.txt           # k x m matrix: coffee-machine power, same shape
```

The script was developed in **Google Colab** and reads these files from a Drive
folder; edit the `DATA_DIR` path near the top to point at wherever the four files
live. (Any local `data/` folder is git-ignored.)

## Notes

Started as a university neural-networks assignment; cleaned up and documented as
a standalone portfolio project.
