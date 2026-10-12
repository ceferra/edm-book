import hashlib
import io
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Statlog German credit (UCI), from a mirror pinned to a commit and checked by its hash
URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
       "d20fcb6402ae34e653d4513b00f39257bb37ed7f/german.csv")
SHA256 = "ec12a88b9fc14d74ba646ea0410cf7ff4533bec2eb61652f8ad76796bbfec017"
raw = urlopen(URL).read()
assert hashlib.sha256(raw).hexdigest() == SHA256, "the data file has changed"
data = pd.read_csv(io.BytesIO(raw), header=None)    # 1000 applicants
y = (data[20] == 2).astype(int)             # 1 = the applicant turned out to be bad
X = data.drop(columns=20)
categorical = X.select_dtypes(exclude="number").columns     # coded as A11, A12...
model = make_pipeline(
    make_column_transformer((OneHotEncoder(handle_unknown="ignore"), categorical),
                            remainder=StandardScaler()),
    LogisticRegression(max_iter=1000))
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, stratify=y,
                                          random_state=1)
p_bad = model.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]

# 1. Is it good? (Unit 1)
print(f"accuracy {np.mean((p_bad >= 0.5) == y_te):.3f}; "
      f"approving everybody: {1 - y_te.mean():.3f}")

# 2. Good for what? Approving a bad applicant costs 5, rejecting a good one 1 (Unit 1)
def cost(threshold):
    reject = p_bad >= threshold
    return (5 * np.sum(~reject & (y_te == 1)) + np.sum(reject & (y_te == 0))) / len(y_te)
print(f"cost per applicant: {cost(0.5):.3f} with threshold 0.5, "
      f"{cost(1 / 6):.3f} with threshold 1/6")

# 3. Good for whom? (Unit 3)
young = (X_te[12] < 25).to_numpy()          # column 12 is the age
approved = p_bad < 1 / 6
print(f"approved: {approved[young].mean():.0%} of the applicants under 25, "
      f"{approved[~young].mean():.0%} of the others")

# 4. Good for how long? If bad applicants become more frequent (Unit 6)
bad_ok = np.mean(p_bad[y_te == 1] < 1 / 6)          # share of bad applicants approved
good_ok = np.mean(p_bad[y_te == 0] < 1 / 6)         # share of good applicants approved
for share in (0.30, 0.50):
    defaults = share * bad_ok / (share * bad_ok + (1 - share) * good_ok)
    print(f"if {share:.0%} of the applicants are bad, {defaults:.1%} of the approved "
          f"loans default")
