import datasets

balanced = datasets.load_from_disk("data/tokenized_datasets/balanced")
unbalanced = datasets.load_from_disk("data/tokenized_datasets/unbalanced")

print(f"Balanced num rows: {balanced.num_rows['train'] + balanced.num_rows['test']}" )
print(f"Unbalanced num rows: {unbalanced.num_rows['train'] + unbalanced.num_rows['test']}" )