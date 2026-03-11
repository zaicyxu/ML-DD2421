from dtree import buildTree, check
from monkdata import monk1, monk2, monk3, monk1test, monk2test, monk3test, attributes

# Build full decision trees
tree_monk1 = buildTree(monk1, attributes)
tree_monk2 = buildTree(monk2, attributes)
tree_monk3 = buildTree(monk3, attributes)

# Evaluate training performance
train_error_monk1 = 1 - check(tree_monk1, monk1)
train_error_monk2 = 1 - check(tree_monk2, monk2)
train_error_monk3 = 1 - check(tree_monk3, monk3)

# Evaluate test performance
test_error_monk1 = 1 - check(tree_monk1, monk1test)
test_error_monk2 = 1 - check(tree_monk2, monk2test)
test_error_monk3 = 1 - check(tree_monk3, monk3test)
