import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DOCS = {   # the knowledge base of a (fictional) bank's assistant
    "card-lost": "If your card is lost or stolen, block it at once in the app or call "
                 "the 24-hour line. A replacement card arrives within five working days.",
    "pin": "You can view the PIN of your card in the app after confirming your "
           "identity. To change it, use any of our cash machines.",
    "limits": "Transfers made in the app are limited to 3,000 euros per day. The limit "
              "can be raised at a branch or with your digital certificate.",
    "abroad": "International transfers outside the euro area take one to three working "
              "days and cost 0.3% of the amount, with a minimum of 6 euros.",
    "overdraft": "An overdraft happens when you spend more than your balance. Overdrawn "
                 "amounts pay 7% annual interest and a one-off fee of 15 euros.",
    "mortgage": "Our fixed-rate mortgage has an APR of 2.9% for up to 30 years and "
                "finances up to 80% of the appraised value of the home.",
    "account": "Opening an account takes ten minutes in the app: you need your identity "
               "document and a selfie. Customers under 25 pay no maintenance fees.",
    "login": "If you cannot log in to the app, update it to the latest version and reset "
             "your access code from the login screen with the code we send by SMS.",
    "savings": "The savings account pays 2% annual interest on balances up to 50,000 "
               "euros, and money can be withdrawn at any time without penalties.",
    "complaints": "Complaints can be filed at any branch, in the app or by post. We "
                  "answer within 15 working days.",
    "phishing": "We never ask for your passwords or codes by phone or e-mail. If you "
                "receive such a message, forward it to our security team.",
    "loan": "Personal loans can be repaid early at any time, with a fee of 1% of the "
            "amount repaid.",
}
QUESTIONS = [("someone stole my wallet with my debit card", "card-lost"),
             ("I forgot the secret number of my card", "pin"),
             ("how much money can I send per day from my phone", "limits"),
             ("what does it cost to send money to the United States", "abroad"),
             ("what happens if my balance goes negative", "overdraft"),
             ("which interest rate do you offer for buying a house", "mortgage"),
             ("what do I need to become a customer", "account"),
             ("the application does not let me sign in", "login"),
             ("how much do I earn if I keep my money with you", "savings"),
             ("I want to complain about a charge", "complaints"),
             ("I got an email asking for my password, is it you?", "phishing"),
             ("can I pay back my loan before the end", "loan")]
OUT_OF_SCOPE = ["what will the weather be like tomorrow", "recommend me a good film"]
ids, texts = list(DOCS), list(DOCS.values())
questions, truth = [q for q, _ in QUESTIONS], [ids.index(d) for _, d in QUESTIONS]

def similarities(vectorizer, queries):
    D = vectorizer.fit_transform(texts)
    return (vectorizer.transform(queries) @ D.T).toarray()   # cosine (rows normalised)

words = TfidfVectorizer(stop_words="english")
chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
retrievers = {"words": lambda q: similarities(words, q),
              "character n-grams": lambda q: similarities(chars, q),
              "both (average)": lambda q: (similarities(words, q)
                                           + similarities(chars, q)) / 2}
print("retriever            recall@1  recall@3  mean reciprocal rank")
for name, retrieve in retrievers.items():
    rank = np.array([list(np.argsort(-s)).index(t) + 1
                     for s, t in zip(retrieve(questions), truth)])
    print(f"{name:20s} {np.mean(rank == 1):8.2f}  {np.mean(rank <= 3):8.2f}  "
          f"{np.mean(1 / rank):10.2f}")

S = retrievers["both (average)"](questions + OUT_OF_SCOPE)
best = S.max(axis=1)
print(f"best score: in-scope questions {best[:12].min():.2f} to {best[:12].max():.2f}, "
      f"out of scope {best[12:].min():.2f} to {best[12:].max():.2f}")
for q, s, t in zip(questions, S, truth):
    if s.argmax() != t:                      # the top passage is not the right one
        print(f"  top-1 miss: '{q}' -> {ids[s.argmax()]}")
