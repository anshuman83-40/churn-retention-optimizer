# ChurnUpliftNet: the model written from scratch

The default model in this project is **not a ready-made library model**. It is a neural network implemented
by hand in NumPy, in one file: [`src/churn/ownmodel.py`](../src/churn/ownmodel.py).

Nothing in it comes from scikit-learn, XGBoost, PyTorch or TensorFlow. NumPy supplies only arrays and
matrix multiplication. The library models (XGBoost, logistic regression, S/T/X-learners) are kept in the
project purely as a **comparison**.

## What was written by hand

| Part | What it does | Where |
|---|---|---|
| Feature encoder | Standardises numbers, one-hot encodes categories | `Encoder` |
| Network layers | Matrix multiply + ReLU, sigmoid output | `forward` |
| Loss | Binary cross-entropy + L2 penalties | `loss_and_grads` |
| **Backpropagation** | Exact gradient of the loss for every weight, derived with the chain rule | `loss_and_grads` |
| Optimiser | Adam (momentum + adaptive step size) | `Adam` |
| Training loop | Mini-batches, shuffling, early stopping on a validation split | `train_one` |
| Ensemble | Several networks with different seeds, averaged | `ChurnUpliftNet.fit` |
| Effect calibration | 2-parameter Newton's method fit | `_calibrate_effect` |
| Explanations | Occlusion: remove one feature, measure the change | `ChurnUpliftNet.explain` |

## The design: one network, two answers

Most projects train one model for "will this customer churn?" and a second for "will a discount help?".
ChurnUpliftNet answers both with one network.

```
customer features (54 numbers after encoding)
        │
   shared trunk: 32 neurons → 16 neurons        "what kind of customer is this?"
        │
        ├── base head  (8 neurons → 1)  →  z0      churn risk if we do nothing
        │
        └── effect head (8 neurons → 1) →  delta   how the offer shifts that risk

P(churn | no offer) = sigmoid(z0)
P(churn | offer)    = sigmoid(z0 + delta)
uplift              = P(churn | no offer) − P(churn | offer)
```

Three design decisions, and why:

1. **The offer effect is a correction on top of the baseline** (`z0 + delta`), not a separate prediction.
   Most of what determines churn is the same with or without an offer, so the two outputs share almost
   everything and only the small difference has to be learned.
2. **The effect head starts at exactly zero and is penalised 10x harder** than the rest. The model begins
   by assuming "the offer does nothing" and only moves away from that when the data insists. This stops it
   inventing effects from noise.
3. **A calibration step undoes the side effect of (2).** The heavy penalty also shrinks real effects, so the
   model first underestimated them (average predicted uplift 1.5 points vs a true 4.0). After training, two
   numbers (a shift and a scale for `delta`) are fitted on the customers who received the offer. That raised
   the value of its recommendations on test data from $9.0k to $30.7k.

## How we know the maths is right

Backpropagation written by hand is easy to get subtly wrong. The test
[`test_backprop_matches_numerical_gradient`](../tests/test_ownmodel.py) nudges every single weight up and
down by a tiny amount, measures how the loss changes, and compares that with what the backpropagation code
says. They agree to a relative error of about 0.0000003. The test runs on every push.

Other tests check that the model learns churn (AUC > 0.80 on a small sample), that the effect head stays at
zero when no customer was treated, and that it finds an offer effect that was deliberately planted in the data.

## Results against the library models

Same train/test splits, same evaluation code. Settings for the network were fixed before looking at the
test results.

**Churn prediction** (1,409 held-out customers)

| Model | ROC-AUC | PR-AUC | Brier (lower is better) |
|---|---|---|---|
| **ChurnUpliftNet (own)** | 0.846 | 0.656 | 0.136 |
| XGBoost (library) | 0.847 | 0.668 | 0.135 |
| Logistic Regression (library) | 0.845 | 0.651 | 0.166 |

**Uplift targeting** (2,113 held-out customers, net value of the offers each model recommends)

| | ChurnUpliftNet (own) | Best library model (S-learner) | Risk targeting |
|---|---|---|---|
| Realistic scenario, offer if value > 0 | $30.7k | $33.7k | $28.9k |
| Realistic scenario, top 20% | $24.3k | $31.0k | $12.9k |
| Risk-aligned scenario, offer if value > 0 | $114.4k | $114.9k | $116.6k |

**Honest reading:** on churn prediction the from-scratch network is level with XGBoost. On uplift it ties
the library model in one scenario and is somewhat behind in the other, while still beating simple risk
targeting. That is the expected pattern: on small tables (7,000 rows) tree models are hard to beat, and the
simulated offer effect follows sharp rules that trees capture easily.

## Limitations

- It rarely flags "sleeping dogs" (customers an offer would push away). Their true effect is about 2 points,
  below what the model can detect with this much data. They are still not sent offers.
- Two-headed networks for treatment effects exist in research (TARNet, DragonNet). This is an independent
  implementation of that idea with its own additive design and calibration step, not a new algorithm.
- The offer experiment is simulated on top of real customers (see the README), so the uplift numbers show how
  well the model recovers a known effect, not a real campaign result.

## Questions you may be asked

**Why is this "your own model" if it uses NumPy?**
NumPy only multiplies matrices. Every modelling decision and every formula (layers, loss, gradients,
optimiser, training, calibration, explanations) is written out in `ownmodel.py`.

**Why not just use XGBoost, since it scores slightly higher?**
The goal was to build and understand a model end to end. The result shows a hand-written network can match
a mature library on churn here, and it gives one model for both tasks instead of two.

**What is backpropagation doing?**
It works backwards from the prediction error. For each weight it computes how much the error would change if
that weight changed slightly, and Adam then moves each weight a small step in the direction that reduces it.

**Why an ensemble of 7 networks?**
A single small network changes noticeably with its random starting weights. Averaging seven evens that out,
which matters most for the uplift estimate because it is a small difference between two predictions.

**How does the explanation work without SHAP?**
For each feature, replace the customer's value with the average customer's value and re-run the network. If
the churn score drops a lot, that feature was pushing this customer toward churn.
