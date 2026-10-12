import joblib
from shiny import App, render, ui
from loan_model import PATH, answer_text

model = joblib.load(PATH)                  # a logistic regression trained beforehand

app_ui = ui.page_fluid(
    ui.panel_title("Loan application"),
    ui.input_slider("income", "Income (thousand EUR)", min=10, max=100, value=30),
    ui.input_slider("debt", "Debt ratio", min=0, max=1, value=0.3, step=0.05),
    ui.output_text("answer"),
)

def server(input, output, session):
    @render.text                           # recomputed whenever an input changes
    def answer():
        return answer_text(model, input.income(), input.debt())

app = App(app_ui, server)                  # run it with: shiny run shiny_app.py
