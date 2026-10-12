import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(19)
TOPICS = {   # what users ask a bank's assistant, by topic
 "balance": ["what is my balance", "how much money do I have", "show my last movements",
             "why is my balance lower than expected"],
 "card": ["my card was blocked", "how do I activate my new card", "card payment declined",
          "I lost my card"],
 "transfer": ["how do I make a transfer", "transfer not received", "cancel a transfer",
              "transfer limit per day"],
 "mortgage": ["mortgage interest rate", "can I repay my mortgage early",
              "documents for a mortgage"],
 "login": ["I cannot log in", "reset my access code", "the app does not open"],
 "bizum": ["how do I send money with bizum", "bizum payment not received",
           "bizum limit per day", "register my phone in bizum"],   # a new service
}
EXTRAS = ["", "please", "hello", "urgent", "thanks", "again", "today"]

def week(n, mix):
    topics, probs = list(mix), np.array(list(mix.values())) / sum(mix.values())
    out = []
    for t in rng.choice(topics, n, p=probs):
        out.append(f"{rng.choice(EXTRAS)} {rng.choice(TOPICS[t])} {rng.choice(EXTRAS)}")
    return out

usual = {"balance": 30, "card": 25, "transfer": 25, "mortgage": 10, "login": 10}
launch = dict(usual, bizum=15)                 # the bank launches instant payments
reference = week(3000, usual)
print("current week compared with the reference   domain AUC   most typical words now")
for name, current in [("same topics", week(3000, usual)),
                      ("new service launched", week(3000, launch))]:
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(reference + current)
    d = np.r_[np.zeros(len(reference)), np.ones(len(current))]
    auc = cross_val_score(LogisticRegression(max_iter=1000), X, d, cv=5,
                          scoring="roc_auc").mean()
    coef = LogisticRegression(max_iter=1000).fit(X, d).coef_[0]
    top = tfidf.get_feature_names_out()[np.argsort(-coef)[:3]]
    print(f"{name:40s}   {auc:8.3f}     {', '.join(top) if auc > 0.55 else '-'}")
