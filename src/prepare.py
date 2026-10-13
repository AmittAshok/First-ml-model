from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/student_scores.csv")
TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)["train"]

df = pd.read_csv(DATA_PATH)

required_columns = {"hours_studied", "score"}
if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Dataset must contain columns: {required_columns}"
    )

if df[list(required_columns)].isnull().any().any():
    raise ValueError("Dataset contains missing values.")

X = df[["hours_studied"]]
y = df["score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["test_size"],
    random_state=params["random_state"],
)

train_df = X_train.copy()
train_df["score"] = y_train

test_df = X_test.copy()
test_df["score"] = y_test

TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print(f"Training rows: {len(train_df)}")
print(f"Testing rows: {len(test_df)}")
print("Data preparation completed.")
