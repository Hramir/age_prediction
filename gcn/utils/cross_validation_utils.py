# Assuming you have your features (X) and labels (y) loaded

# Create a function to perform k-fold cross-validation
def kfold_cross_validation(X, y, model, k=5):
    n = len(X)
    fold_size = n // k

    # Shuffle indices
    indices = list(range(n))
    import random
    random.shuffle(indices)

    scores = []

    for i in range(k):
        # Define the indices for the current fold
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < k - 1 else n
        test_indices = indices[start_idx:end_idx]
        train_indices = indices[:start_idx] + indices[end_idx:]

        # Split the data into training and testing sets
        X_train = [X[idx] for idx in train_indices]
        y_train = [y[idx] for idx in train_indices]
        X_test = [X[idx] for idx in test_indices]
        y_test = [y[idx] for idx in test_indices]

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        accuracy = model.score(X_test, y_test)
        scores.append(accuracy)

        print(f'Fold {i + 1}: Accuracy = {accuracy}')

    mean_accuracy = sum(scores) / k
    print(f'Mean Accuracy: {mean_accuracy}')

# Example usage
from sklearn.ensemble import RandomForestClassifier  # Replace with your desired model

# Assuming you have your features (X) and labels (y) loaded
# Replace RandomForestClassifier with your desired model
model = RandomForestClassifier()

# Replace X and y with your actual data
kfold_cross_validation(X, y, model, k=5)
