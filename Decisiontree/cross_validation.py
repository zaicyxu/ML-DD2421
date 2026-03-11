import random
from dtree import buildTree, check, allPruned
from monkdata import monk1, attributes

# Split dataset into k folds
def cross_validation_splits(dataset, k=10):
    shuffled = list(dataset)
    random.shuffle(shuffled)
    fold_size = len(dataset) // k
    return [shuffled[i * fold_size:(i + 1) * fold_size] for i in range(k)]

# Perform cross-validation pruning
def cross_validate_pruning(dataset, attributes, k=10):
    folds = cross_validation_splits(dataset, k)
    best_pruned_tree = None
    lowest_avg_error = float("inf")

    for fold_index in range(k):
        # Training set: all but one fold
        training_data = [sample for i, fold in enumerate(folds) if i != fold_index for sample in fold]
        # Validation set: the remaining fold
        validation_data = folds[fold_index]

        # Train full tree
        tree = buildTree(training_data, attributes)

        # Try pruning and check validation accuracy
        best_tree = tree
        best_error = 1 - check(tree, validation_data)

        for pruned_tree in allPruned(tree):
            pruned_error = 1 - check(pruned_tree, validation_data)
            if pruned_error < best_error:
                best_tree = pruned_tree
                best_error = pruned_error

        # Track the best pruned tree
        if best_error < lowest_avg_error:
            best_pruned_tree = best_tree
            lowest_avg_error = best_error

    return best_pruned_tree

# Apply cross-validation pruning
best_tree_monk1 = cross_validate_pruning(monk1, attributes)

# Evaluate the pruned tree
error_monk1 = 1 - check(best_tree_monk1, monk1)
print(f"MONK-1 Test Error After Cross-Validation Pruning: {error_monk1:.4f}")
