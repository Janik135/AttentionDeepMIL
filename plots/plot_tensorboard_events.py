from glob import glob
from tensorboard.backend.event_processing import event_accumulator as ea
import os

dir_path = "/Users/janik/Documents/TU Darmstadt/21 SS/MA/Experimente/Gerste/5_1000_dropout0_02_7_3/"
subfolders = glob(os.path.join(dir_path, "*/"))

for folder in subfolders:
    acc = ea.EventAccumulator(folder)
    acc.Reload()

    # Print tags of contained entities, use these names to retrieve entities as below
    print(acc.Tags())

    train_loss = [(s.step, s.value) for s in acc.Scalars('Loss/train')]
    train_bal_accuracy = [(s.step, s.value) for s in acc.Scalars('Accuracy/train')]
    test_loss = [(s.step, s.value) for s in acc.Scalars('Loss/test')]
    test_bal_accuracy = [(s.step, s.value) for s in acc.Scalars('Accuracy/test')]
