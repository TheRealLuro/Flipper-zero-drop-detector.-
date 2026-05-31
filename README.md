# Flipper Zero Drop Detector

This project teaches a Flipper Zero to notice how it's being moved and to sound an alarm the moment it gets dropped. It tells four kinds of motion apart: sitting still, being carried while walking, being handled, and falling.

The Flipper's motion sensor only gives you raw numbers, so we trained a small neural network to read them. The work splits into three parts:

1. A data collector that runs on the Flipper and saves labeled motion to the SD card.
2. A trainer on the PC that turns those recordings into a model.
3. A detector that runs the trained model back on the Flipper and sets off a siren when it sees a drop.

## What's in here

```
flipper-zero/
  expo_app/      the FliPort data collector (records training data)
  drop-dect/     the live drop detector (app + background service)
model/           PC training code and the desktop workbench
MODEL_CARD.md    the model: data, architecture, accuracy, and limits
```

## The model

It's a small 1-D convolutional network, about 268 numbers in total (roughly 1 KB). We wrote the whole thing by hand in C++, no PyTorch or TensorFlow, so it stays small enough to run on the device. On held-out test data it gets about 90% of windows right and, more importantly, caught every drop in testing. The weak spot is telling "idle" from light fidgeting. None of that matters for the alarm, since only the drop class triggers it. Full details are in [MODEL_CARD.md](MODEL_CARD.md).

## Running it

Each part has its own short guide:

- Collect data on the Flipper: [flipper-zero/expo_app/README.md](flipper-zero/expo_app/README.md)
- Train the model on a PC: [model/readme.md](model/readme.md)
- Run the detector on the Flipper: [flipper-zero/drop-dect/README.md](flipper-zero/drop-dect/README.md)

If you just want to see the detector go, the quickest path is:

```powershell
cd flipper-zero\drop-dect
ufbt launch
```

Then open `Apps > Tools > FliPort Live` on the device and drop it on something soft.

## Status and limits

The model trains and runs end to end, and the detector works as both a regular app and a background service. A few honest caveats:

- The training set is small and came from one person on one device, so it may not generalize to everyone.
- The drops it learned from are short, deliberate test drops, not real-world falls.
- "Idle" recordings are scarce, so still windows often get read as fidgeting. Again, harmless for the alarm.
