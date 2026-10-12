import pandas as pd

items = ["Cabezon", "Chispa", "TioPaco", "Xufer", "Goloso", "Trigo", "LaVaca"]
T = pd.DataFrame([[1, 1, 0, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0],
                  [1, 1, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1, 1],
                  [0, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 1], [1, 1, 0, 0, 1, 0, 1],
                  [0, 1, 0, 0, 1, 0, 0]], columns=items).astype(bool)

A = T["Goloso"] & T["Xufer"]          # antecedent: {Goloso, Xufer}
B = T["Trigo"]                        # consequent: {Trigo}
supp = (A & B).mean()
conf = supp / A.mean()
lift = conf / B.mean()
leverage = supp - A.mean() * B.mean()
conviction = (1 - B.mean()) / (1 - conf)
print(f"support={supp:.2f} confidence={conf:.2f} lift={lift:.2f} "
      f"leverage={leverage:.2f} conviction={conviction:.2f}")
