import pandas as pd
import numpy as np
import mlxtend
from skmultilearn.problem_transform import LabelPowerset








df = pd.read_excel(r'C:\Users\Asus\Documents\Final_Exam_DL\pythonProject1\Exam2-v5\excel\train_test.xlsx')

dfx = df[df['split'] == 'train']


def count_label_occurrences(df):
    label_counts = {}
    for _, row in df.iterrows():
        target_labels = row['target'].split(',')  # Split target labels
        target_classes = list(map(int, row['target_class'].split(',')))  # Split target class and convert to int
        for label, cls in zip(target_labels, target_classes):
            label_counts[label] = label_counts.get(label, {'0': 0, '1': 0})  # Initialize counts for each label
            label_counts[label][str(cls)] += 1  # Increment count for 0 or 1
    return label_counts

# Call the function
label_counts = count_label_occurrences(dfx)

class_weights = {}
positive_weights = {}
negative_weights = {}

for label, counts in label_counts.items():
    num_1s = counts.get('1', 0)
    num_0s = counts.get('0', 0)
    total_samples = num_1s + num_0s

    positive_weights[label] = total_samples / (2 * num_1s) if num_1s != 0 else 0
    negative_weights[label] = total_samples / (2 * num_0s) if num_0s != 0 else 0

class_weights['positive_weights'] = positive_weights
class_weights['negative_weights'] = negative_weights

# print(class_weights)

from skmultilearn.problem_transform import LabelPowerset


labels = list(label_counts.keys())

# Create the LabelPowerset transformer
lp = LabelPowerset()

# Fit and transform the label counts data to the LabelPowerset format
transformed_labels = lp.transform(labels)

# Print the transformed labels
print(transformed_labels)
# Print the results
# for label, counts in label_counts.items():
#     print(f"Target: {label} - Count of 0s: {counts['0']}, Count of 1s: {counts['1']}")








