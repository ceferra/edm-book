from shiny import App, render, ui

RUNS = {"top level": 0, "server": 0, "render": 0}        # how often each part runs

RUNS["top level"] += 1             # runs once, when the application starts:
CATALOGUE = list(range(1, 1001))   # load data, models and libraries here

app_ui = ui.page_fluid(ui.input_slider("n", "Products", 1, 1000, 10),
                       ui.output_text("total"))

def server(input, output, session):
    RUNS["server"] += 1            # runs once per user session:
    basket = []                    # objects that each user needs their own copy of

    @render.text
    def total():
        RUNS["render"] += 1        # runs every time an input it reads changes
        return f"Total of the first {input.n()} products: {sum(CATALOGUE[:input.n()])}"

app = App(app_ui, server)
