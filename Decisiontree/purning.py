from dtree import buildTree, check, allPruned
from monkdata import monk1, monk3, monk1test, monk3test, attributes

# 1. Train the original trees
tree_monk1 = buildTree(monk1, attributes)
tree_monk3 = buildTree(monk3, attributes)

# 2. Compute initial test errors
error_monk1_before = 1 - check(tree_monk1, monk1test)
error_monk3_before = 1 - check(tree_monk3, monk3test)

# 3. Perform pruning
best_tree_monk1 = tree_monk1
best_tree_monk3 = tree_monk3

for pruned_tree in allPruned(tree_monk1):
    if 1 - check(pruned_tree, monk1test) < error_monk1_before:
        best_tree_monk1 = pruned_tree
        error_monk1_before = 1 - check(pruned_tree, monk1test)

for pruned_tree in allPruned(tree_monk3):
    if 1 - check(pruned_tree, monk3test) < error_monk3_before:
        best_tree_monk3 = pruned_tree
        error_monk3_before = 1 - check(pruned_tree, monk3test)

# 4. Compute post-pruning test errors
error_monk1_after = 1 - check(best_tree_monk1, monk1test)
error_monk3_after = 1 - check(best_tree_monk3, monk3test)

# 5. Print results
print(f"MONK-1 Test Error Before Pruning: {error_monk1_before:.4f}")
print(f"MONK-1 Test Error After Pruning: {error_monk1_after:.4f}")

print(f"MONK-3 Test Error Before Pruning: {error_monk3_before:.4f}")
print(f"MONK-3 Test Error After Pruning: {error_monk3_after:.4f}")

