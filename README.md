### Confusion Matrix and Classification Report

This repository includes a demonstration of generating a confusion matrix and a classification report using Python libraries such as NumPy, scikit-learn, seaborn, and matplotlib.

---

#### Overview

The provided code showcases the creation of a confusion matrix and a classification report to evaluate the performance of a classification model. It compares actual and predicted labels for a binary classification task involving the categories 'Dog' and 'Not Dog'.

---

#### Implementation Details

1. **Confusion Matrix**: 
   - The confusion matrix is created using `confusion_matrix` function from scikit-learn (`sklearn.metrics`).
   - It visualizes the count of true positive, true negative, false positive, and false negative predictions.
   - The matrix is displayed as a heatmap using `seaborn` and `matplotlib.pyplot`.

2. **Classification Report**:
   - The classification report is generated using `classification_report` function from scikit-learn.
   - It provides precision, recall, F1-score, and support metrics for each class ('Dog' and 'Not Dog').
   - The report summarizes the performance of the classifier in terms of accuracy and per-class metrics.

---

#### Example Usage

```python
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Actual and predicted labels
actual = np.array(['Dog', 'Dog', 'Dog', 'Not Dog', 'Dog', 'Not Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'])
predicted = np.array(['Dog', 'Not Dog', 'Dog', 'Not Dog', 'Dog', 'Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'])

# Compute confusion matrix
cm = confusion_matrix(actual, predicted)

# Plot confusion matrix as heatmap
sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=['Dog', 'Not Dog'],
            yticklabels=['Dog', 'Not Dog'])
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Confusion Matrix', fontsize=17)
plt.show()

# Print classification report
print(classification_report(actual, predicted))
