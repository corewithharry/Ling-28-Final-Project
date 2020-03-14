"""
Testing for the validation cases of Joy, Anger, and Sad.
"""

import numpy as np
from classifier import classify
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def is_correct(results, label):
    minimum = float

VALIDATION_ANGER = open("datasets/testing/validation_anger.txt").read().split("\n")
VALIDATION_SAD = open("datasets/testing/validation_sad.txt").read().split("\n")
VALIDATION_JOY = open("datasets/testing/validation_joy.txt").read().split("\n")

truth = ["#-ANGER", "#-ANGER", "#-ANGER", "#-ANGER", "#-ANGER", "#-ANGER", "#-ANGER", "#-ANGER", "#-ANGER", "#-ANGER", 
          "#-SAD", "#-SAD", "#-SAD", "#-SAD", "#-SAD", "#-SAD", "#-SAD", "#-SAD", "#-SAD", "#-SAD",
          "#-JOY", "#-JOY", "#-JOY", "#-JOY", "#-JOY", "#-JOY", "#-JOY", "#-JOY", "#-JOY", "#-JOY"]

predictions = []

"""for phrase in VALIDATION_ANGER:
    print("Classifying Anger:")
    score, label = classify(phrase)
    predictions.append(label)

print("==========================")

for phrase in VALIDATION_SAD:
    print("Classifying Sad:")
    score, label = classify(phrase)
    predictions.append(label)
print("==========================")

for phrase in VALIDATION_JOY:
    print("Classifying Joy:")
    score, label = classify(phrase)
    predictions.append(label)
print("==========================")"""

predictions = ["#-ANGER", "#-ANGER", "#-ANGER", "#-ANGER", "#-ANGER", "#-ANGER", "#-ANGER", "#-ANGER", "#-JOY", "#-ANGER", "#-ANGER", "#-JOY", "#-SAD", "#-SAD", "#-ANGER", "#-SAD", "#-SAD", "#-SAD", "#-SAD", "#-JOY", "#-JOY", "#-JOY", "#-JOY", "#-JOY", "#-JOY", "#-JOY", "#-JOY", "#-ANGER", "#-SAD", "#-JOY"]

names = ["#-JOY", "#-ANGER", "#-SAD"]

cm = confusion_matrix(truth, predictions, labels=names)
report = classification_report(truth, predictions, labels=names)

print(cm)
print(predictions)
print(report)