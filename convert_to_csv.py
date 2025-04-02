import pandas as pd

# Read the raw text file
with open("reviews.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Parse lines into a DataFrame
data = []
for line in lines[:1000]:  # Limit to 1000 for demo
    if line.startswith("__label__"):
        label = int(line.split()[0].replace("__label__", ""))
        review_text = " ".join(line.split()[1:]).strip()
        data.append({
            "label": label,  # 1 or 2 (negative or positive)
            "review_text": review_text,
            "product": "Unknown Product",  # Placeholder
            "rating": 5 if label == 2 else 1  # Map label to rating
        })

df = pd.DataFrame(data)
df.to_csv("reviews.csv", index=False)
print("Converted to reviews.csv")
