import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load and split the data
iris = load_iris()
X, y = iris.data, iris.target

# Use stratification to ensure balanced classes in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # This ensures balanced classes
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Define parameter ranges
max_depth_range = list(range(1, 11))
min_samples_split_range = list(range(2, 21, 2))

# Arrays to store results for each parameter combination
train_scores = np.zeros((len(max_depth_range), len(min_samples_split_range)))
test_scores = np.zeros((len(max_depth_range), len(min_samples_split_range)))

# Perform grid search with both parameters together
for i, max_depth in enumerate(max_depth_range):
    for j, min_samples_split in enumerate(min_samples_split_range):
        dt = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        dt.fit(X_train, y_train)

        # Get predictions
        y_train_pred = dt.predict(X_train)
        y_test_pred = dt.predict(X_test)

        # Store scores
        train_scores[i, j] = accuracy_score(y_train, y_train_pred)
        test_scores[i, j] = accuracy_score(y_test, y_test_pred)

# Create plots for hyperparameter tuning
plt.figure(figsize=(12, 5))

# Plot max_depth effects
plt.subplot(1, 2, 1)
avg_train_scores_depth = np.mean(train_scores, axis=1)
avg_test_scores_depth = np.mean(test_scores, axis=1)

plt.plot(max_depth_range, avg_train_scores_depth, label='Training Score', marker='o')
plt.plot(max_depth_range, avg_test_scores_depth, label='Test Score', marker='o')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Impact of max_depth\non Model Performance')
plt.legend()
plt.grid(True)

# Plot min_samples_split effects
plt.subplot(1, 2, 2)
avg_train_scores_split = np.mean(train_scores, axis=0)
avg_test_scores_split = np.mean(test_scores, axis=0)

plt.plot(min_samples_split_range, avg_train_scores_split, label='Training Score', marker='o')
plt.plot(min_samples_split_range, avg_test_scores_split, label='Test Score', marker='o')
plt.xlabel('min_samples_split')
plt.ylabel('Accuracy')
plt.title('Impact of min_samples_split\non Model Performance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
plt.close()

# Find best parameters (using test scores)
best_i, best_j = np.unravel_index(np.argmax(test_scores), test_scores.shape)
best_max_depth = max_depth_range[best_i]
best_min_samples_split = min_samples_split_range[best_j]

print("\nGrid Search Results:")
for i, max_depth in enumerate(max_depth_range):
    for j, min_samples_split in enumerate(min_samples_split_range):
        print(f"max_depth={max_depth}, min_samples_split={min_samples_split}: "
              f"Train={train_scores[i,j]:.3f}, Test={test_scores[i,j]:.3f}, "
              f"Diff={train_scores[i,j]-test_scores[i,j]:.3f}")

print("\nBest Parameters:")
print(f"max_depth: {best_max_depth}")
print(f"min_samples_split: {best_min_samples_split}")
print(f"Best training score: {train_scores[best_i, best_j]:.3f}")
print(f"Best test score: {test_scores[best_i, best_j]:.3f}")
print(f"Difference: {train_scores[best_i, best_j] - test_scores[best_i, best_j]:.3f}")

# Train final model with best parameters
best_model = DecisionTreeClassifier(
    max_depth=best_max_depth,
    min_samples_split=best_min_samples_split,
    random_state=42
)
best_model.fit(X_train, y_train)

# Generate predictions for final model
y_pred = best_model.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Print feature importances
importances = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': best_model.feature_importances_
})
print("\nFeature Importances:")
print(importances.sort_values('importance', ascending=False))

# Visualize the final decision tree
plt.figure(figsize=(20, 10))
plot_tree(best_model,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          proportion=True)
plt.title(f"Decision Tree (max_depth={best_max_depth}, min_samples_split={best_min_samples_split})")
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()