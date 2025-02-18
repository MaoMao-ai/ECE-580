#Data Preparation and Exploration
import torch

# === Step 2: Loading the Dataset === #
def load_dataset(file_path):
    """
    Loads the saved PyTorch .pt dataset from the specified file path.

    This function reads a dataset saved in PyTorch format and extracts its features, target values,
    and relevant metadata. It prints the shapes of the feature matrix and target vector,
    as well as the feature and target names.

    Parameters:
    -----------
    file_path : str
        The file path to the saved .pt dataset.

    Returns:
    --------
    tuple
        A tuple containing:
        - features (torch.Tensor): A tensor containing the feature values.
        - target (torch.Tensor): A tensor containing the target values.
        - feature_names (list of str): A list of names corresponding to the feature columns.
        - target_name (str): The name of the target variable.

    Example:
    --------
    >>> features, target, feature_names, target_name = load_dataset("cleaned_automobile_train_dataset.pt")
    Loaded dataset successfully!
    Features shape: torch.Size([100, 13]), Target shape: torch.Size([100])
    Feature Names: ['wheel-base', 'length', 'width', ...], Target Name: 'price'
    """
    dataset = torch.load(file_path)
    print("Loaded dataset successfully!")
    print(f"Features shape: {dataset['features'].shape}, Target shape: {dataset['target'].shape}")
    print(f"Feature Names: {dataset['feature_names']}, Target Name: {dataset['target_name']}")
    return dataset["features"], dataset["target"], dataset["feature_names"], dataset["target_name"]


if __name__ == "__main__":
    # Load the dataset and display information
    print('loading train_dataset from cleaned_automobile_train_dataset.pt')
    X_train, Y_train, feature_names, target_name = load_dataset("cleaned_automobile_train_dataset.pt")
    print('number of training samples={}'.format(X_train.shape[0]))
    print('dimension of features={}'.format(X_train.shape[1]))
    print('X_train shape={}'.format(list(X_train.shape)))
    print('Y_train shape={}'.format(list(Y_train.shape)))

    print('loading test_dataset from cleaned_automobile_test_dataset.pt')
    X_test, Y_test, feature_names, target_name = load_dataset("cleaned_automobile_test_dataset.pt")
    print('number of test samples={}'.format(X_test.shape[0]))
    print('dimension of features={}'.format(X_test.shape[1]))
    print('X_test shape={}'.format(list(X_test.shape)))
    print('Y_test shape={}'.format(list(Y_test.shape)))