import matplotlib.pyplot as plt
import numpy as np
from shiny import App, reactive, render, ui
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from crm_data import pilot

X_train, y_train, X_val, y_val, X_rest, _, _ = pilot()   # loaded once per Python process
N = len(X_rest)                                          # customers in the campaign
MODELS = {"Decision tree": lambda: DecisionTreeClassifier(max_depth=5, random_state=0),
          "Random forest": lambda: RandomForestClassifier(200, min_samples_leaf=20,
                                                          random_state=0),
          "Nearest neighbours": lambda: make_pipeline(StandardScaler(),
                                                     KNeighborsClassifier(100))}
TRAININGS = []                                           # a log, to see the caching

def profit_curve(ranked, fixed, per_mail, margin, n=N):
    """Profit of not launching (index 0) and of mailing the top k customers, for every k,
    scaled to n customers; the fixed cost is paid only if the campaign is launched."""
    ranked = np.asarray(ranked)
    share = np.arange(1, len(ranked) + 1) / len(ranked)
    mailed = margin * np.cumsum(ranked) * n / len(ranked) - per_mail * share * n - fixed
    return np.r_[0.0, mailed]

def decision(profit, n_validation):
    """The best action: not launching (index 0) or mailing the top customers."""
    best = int(np.argmax(profit))
    if best == 0:
        return "Do not launch the campaign: every mailing loses money"
    return (f"Mail the top {best / n_validation:.0%} of {N:,} customers: "
            f"profit {profit[best]:,.0f} EUR")

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select("model", "Model", list(MODELS)),
        ui.input_slider("fixed", "Initial cost (EUR)", 0, 40_000, 10_000, step=1_000),
        ui.input_slider("per_mail", "Cost per brochure (EUR)", 0.1, 3.0, 1.5, step=0.1),
        ui.input_slider("margin", "Profit per sale (EUR)", 5, 60, 30),
    ),
    ui.output_text("summary"),
    ui.layout_columns(ui.output_plot("response"), ui.output_plot("profit_plot")),
    title="Mailing campaign planner",
)

def server(input, output, session):
    @reactive.calc                     # cached: recomputed only when the model changes
    def ranked():
        TRAININGS.append(input.model())
        model = MODELS[input.model()]().fit(X_train, y_train)
        return y_val[np.argsort(-model.predict_proba(X_val)[:, 1])]

    def profit():
        return profit_curve(ranked(), input.fixed(), input.per_mail(), input.margin())

    @render.text
    def summary():
        return decision(profit(), len(ranked()))

    @render.plot
    def response():
        r = ranked()
        x = np.arange(1, len(r) + 1) / len(r)
        fig, ax = plt.subplots()
        ax.plot(x, np.cumsum(r) / r.sum(), label=input.model())
        ax.plot([0, 1], [0, 1], "--", label="random")
        ax.set(xlabel="share of customers mailed", ylabel="share of responses")
        ax.legend()
        return fig

    @render.plot(alt="Profit curve")
    def profit_plot():
        p = profit()
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(p)) / (len(p) - 1), p)        # 0: not launching
        ax.axhline(0, color="grey")
        ax.set(xlabel="share of customers mailed", ylabel="profit (EUR)")
        return fig

app = App(app_ui, server)                                # run: shiny run crm_app.py
