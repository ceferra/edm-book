import pandas as pd

# Six largest departments, UC Berkeley 1973 (Bickel et al., 1975)
d = pd.DataFrame({"dept": list("ABCDEF"),
                  "men": [825, 560, 325, 417, 191, 373],
                  "men_adm": [512, 353, 120, 138, 53, 22],
                  "women": [108, 25, 593, 375, 393, 341],
                  "women_adm": [89, 17, 202, 131, 94, 24]})
d["men_rate"], d["women_rate"] = d.men_adm / d.men, d.women_adm / d.women
cols = ["dept", "men", "men_rate", "women", "women_rate"]
print(d[cols].round(2).to_string(index=False))
print(f"aggregate:   men {d.men_adm.sum() / d.men.sum():.3f}   "
      f"women {d.women_adm.sum() / d.women.sum():.3f}")
w = (d.men + d.women) / (d.men + d.women).sum()          # department mix of all applicants
men_adj, women_adj = (w * d.men_rate).sum(), (w * d.women_rate).sum()
print(f"adjusted:    men {men_adj:.3f}   women {women_adj:.3f}")
