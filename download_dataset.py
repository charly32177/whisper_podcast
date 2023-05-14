from datasets import load_dataset

dataset = load_dataset("squad_v2")

print(dataset)

context = set()

for data in dataset["train"]:
    context.add(data["context"])

for data in dataset["validation"]:
    context.add(data["context"])

print(len(context))