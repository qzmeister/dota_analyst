with open("business/ml/targets.py") as f:
    data = f.read()
print("triple-quote count:", data.count('"""'))
print("line count:", len(data.split("\n")))
