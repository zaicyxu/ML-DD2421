from dtree import entropy, select, averageGain
from monkdata import monk1, monk2, monk3, attributes

for dataset, name in zip([monk1, monk2, monk3], ["MONK-1", "MONK-2", "MONK-3"]):
    print(f"\nInformation Gain for {name}:")
    for i, attr in enumerate(attributes):
        gain = averageGain(dataset, attr)
        print(f"Attribute a{i+1}: {gain:.4f}")
