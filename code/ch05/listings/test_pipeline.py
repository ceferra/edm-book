import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from pipeline_lib import FEATURES, clip_age, make_data, train, validate

MODEL = train(*make_data(500))

def test_clip_age():                                         # unit test
    df = pd.DataFrame({"age": [10, 50, 130], "income": [20.0, None, 40.0], "debt": 1.0})
    assert clip_age(df)["age"].tolist() == [18, 50, 100]

def test_imputation_learned_once():                         # unit test: no batch effects
    alone = pd.DataFrame({"age": [40], "income": [np.nan], "debt": [5.0]})
    others = pd.DataFrame({"age": [30, 50], "income": [100.0, 200.0],
                           "debt": [1.0, 2.0]})
    p_alone = MODEL.predict_proba(alone)[0, 1]
    p_batch = MODEL.predict_proba(pd.concat([alone, others]))[0, 1]
    assert np.isclose(p_alone, p_batch)
    df, _ = make_data(500)
    imputer = MODEL.named_steps["simpleimputer"]
    assert np.isclose(imputer.statistics_[1], df.income.median())

rows = st.lists(st.tuples(st.integers(-50, 200), st.none() | st.floats(0, 1e6),
                          st.floats(0, 50)), min_size=1, max_size=50)

@given(rows)
def test_predictions_are_probabilities(rows):               # property-based test
    p = MODEL.predict_proba(pd.DataFrame(rows, columns=FEATURES))[:, 1]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()

def test_validate_detects_bad_data():                       # data validation
    df, _ = make_data(50)
    assert validate(df) == []
    assert validate(df.assign(income=-df.income.abs())) == ["negative income"]
    assert validate(df.assign(debt=np.inf)) == ["infinite values"]
    assert validate(df.drop(columns="debt")) == ["missing column debt"]

def test_model_beats_baseline():                            # model quality
    df, y = make_data(2000)
    model = train(df[:1500], y[:1500])
    accuracy = model.score(df[1500:], y[1500:])
    assert accuracy > max(y[1500:].mean(), 1 - y[1500:].mean()) + 0.05

def test_invariance_to_identifiers():                       # invariance test
    df, _ = make_data(500)
    renamed = df.assign(name="someone else")
    assert np.allclose(MODEL.predict_proba(df), MODEL.predict_proba(renamed))

def test_more_income_never_hurts():                         # directional expectation
    x = make_data(100)[0]
    richer = x.assign(income=x.income + 10)
    assert (MODEL.predict_proba(richer)[:, 1] >= MODEL.predict_proba(x)[:, 1]).all()

def test_pipeline_end_to_end():                             # integration test
    df, y = make_data(300, seed=1)
    df.loc[::10, "income"] = None                           # missing values in new data
    assert validate(df) == []
    predictions = train(df, y).predict(df)
    assert len(predictions) == len(df) and set(predictions) <= {0, 1}
