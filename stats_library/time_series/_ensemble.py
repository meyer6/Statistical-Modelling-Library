import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame, date_range

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

class Forecaster:
    def fit(self, y, **kwargs):
        raise NotImplementedError

    def predict(self, step=1):
        raise NotImplementedError

class ETSForecaster(Forecaster):
    def __init__(self, seasonal_periods=None, trend='add', seasonal=None):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.fitted_ = None

    def fit(self, y, **kwargs):
        model = ExponentialSmoothing(y, trend=self.trend, seasonal=self.seasonal,
                                     seasonal_periods=self.seasonal_periods)
        self.fitted_ = model.fit(optimized=True)
        return self

    def predict(self, step=1):
        if self.fitted_ is None:
            raise RuntimeError('Call fit(...) before predict(...)')
        
        forecast = self.fitted_.forecast(step)
        return np.asarray(forecast, dtype=float)

class SARIMAXForecaster(Forecaster):
    def __init__(self, order=(1,0,1), seasonal_order=(0,0,0,0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_ = None

    def fit(self, y, **kwargs):
        model = SARIMAX(y,
                        order=self.order, seasonal_order=self.seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        
        self.fitted_ = model.fit(disp=False)

        return self

    def predict(self, step=1):
        if self.fitted_ is None:
            raise RuntimeError('Call fit(...) before predict(...)')

        res = self.fitted_.get_forecast(steps=step)
        return np.asarray(res.predicted_mean, dtype=float)

class ProphetForecaster(Forecaster):
    def fit(self, y, dates=None, freq='MS'):
        if not PROPHET_AVAILABLE:
            raise RuntimeError("Prophet is not available in this environment.")
        self.freq_ = freq
        self.history_len_ = len(y)
        if dates is None and hasattr(y, 'index'):
            dates_used = pd.to_datetime(y.index)
            inferred = pd.infer_freq(dates_used)
            if inferred is not None:
                self.freq_ = inferred
        elif dates is not None:
            dates_used = pd.to_datetime(dates)
        else:
            dates_used = date_range(start='2000-01-01', periods=len(y), freq=self.freq_)
        self.start_ = dates_used[0]
        df = DataFrame({'ds': dates_used, 'y': np.asarray(y, dtype=float)})
        # monthly data -> yearly seasonality
        self.model_ = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        self.model_.fit(df)
        return self

    def predict(self, step=1):
        last_date = self.start_ + pd.tseries.frequencies.to_offset(self.freq_) * (self.history_len_ - 1)
        future_dates = pd.date_range(start=last_date, periods=step + 1, freq=self.freq_)[1:]
        future_df = DataFrame({'ds': future_dates})
        forecast = self.model_.predict(future_df)
        return np.asarray(forecast['yhat'][-step:], dtype=float)

class EnsembleTimeForecaster:
    def __init__(self, models):
        self.models = models
        self.n_models = len(self.models)
        self.weights = np.ones(self.n_models) / self.n_models
        self.last_y = None

    def fit(self, y):
        for model in self.models:
            model.fit(y)

        self.last_y = np.copy(y)

    def predict(self, step=1):
        if self.last_y is None:
            raise ValueError("Cannot predict before fit")
        
        return np.dot(self.weights, [model.predict(step=step) for model in self.models])

    def update(self, next_y):
        if self.last_y is None:
            raise ValueError("Cannot update before fit")
        
        for i in range(len(next_y)):
            losses = np.array([(model.predict(step=1)[0] - next_y[i]) ** 2 for model in self.models])
            losses /= max(losses)

            weight_multipliers = np.exp(-0.5 * losses)
            self.weights = (self.weights + self.weights * weight_multipliers) * 0.5
            self.weights /= sum(self.weights)

        self.last_y = np.append(self.last_y, next_y)

        for model in self.models:
            model.fit(np.array(self.last_y))

        return self

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    # --- Load monthly series (AirPassengers if available, else synthetic) ---
    try:
        # try a couple of loaders (some environments differ)
        try:
            ds = sm.datasets.get_rdataset("iris")
            data = ds.data
            if 'value' in data.columns:
                vals = data['value'].astype(float).values
            else:
                vals = data.select_dtypes(include=[np.number]).iloc[:, 0].astype(float).values
            series = pd.Series(vals, index=pd.date_range("1949-01-01", periods=len(vals), freq="MS"))
        except Exception:
            # fallback to statsmodels loader (might not exist everywhere)
            d2 = sm.datasets.airpassengers.load_pandas()
            ap = d2.data['AirPassengers']
            if isinstance(ap.index, pd.PeriodIndex):
                ap.index = ap.index.to_timestamp("MS")
            series = ap.astype(float)
    except Exception:
        # final fallback: synthetic monthly seasonal series
        print("Could not load AirPassengers; using synthetic monthly series.")
        rng = pd.date_range("2000-01-01", periods=144, freq="MS")
        series = pd.Series(100 + 10*np.sin(2*np.pi*np.arange(144)/12) + np.arange(144)*0.5 + np.random.randn(144)*3,
                           index=rng)

    # --- Split dataset as requested ---
    n = len(series)
    i50 = int(0.5 * n)
    i75 = int(0.75 * n)

    train50 = series.iloc[:i50]
    middle25 = series.iloc[i50:i75]
    train75 = series.iloc[:i75]    # used for individual models
    final25 = series.iloc[i75:]

    h = len(final25)

    print(f"Series length: {n}; train50={len(train50)}, middle25={len(middle25)}, final25={len(final25)}")

    # --- Instantiate your original models (no modification of classes) ---
    # sensible params for monthly series
    ets = ETSForecaster(seasonal_periods=12, trend='add', seasonal='mul')
    sarimax = SARIMAXForecaster(order=(1,1,1), seasonal_order=(1,1,1,12))

    # Prophet: try to use if available (your class may raise if prophet isn't installed)
    use_prophet = False
    try:
        prophet = ProphetForecaster()
        # do not call fit here yet; just acknowledge we can try
        use_prophet = True
    except Exception:
        use_prophet = False

    results = {}

    # --- Fit individual models on first 75% and forecast final 25% ---
    print("Fitting ETS on first 75% ...")
    ets.fit(train75.values)
    results['ETS'] = ets.predict(step=h)

    print("Fitting SARIMAX on first 75% ...")
    sarimax.fit(train75.values)
    results['SARIMAX'] = sarimax.predict(step=h)

    if use_prophet:
        try:
            print("Fitting Prophet on first 75% ...")
            # your original ProphetForecaster.fit may expect either array or series; try series then values
            try:
                prophet.fit(train75)
            except Exception:
                prophet.fit(train75.values)
            results['Prophet'] = prophet.predict(step=h)
        except Exception as e:
            print("Prophet failed during fit/predict; skipping Prophet. Error:", e)
            use_prophet = False

    # --- Ensemble: fit on first 50%, update with middle 25% step-by-step and record weights/preds/losses ---
    ensemble_members = [
        ETSForecaster(seasonal_periods=12, trend='add', seasonal='mul'),
        SARIMAXForecaster(order=(1,1,1), seasonal_order=(1,1,1,12))
    ]
    if use_prophet:
        ensemble_members.append(ProphetForecaster())

    ensemble = EnsembleTimeForecaster(models=ensemble_members)
    ensemble.fit(train50.values)   # use first 50%

    # We'll record arrays externally (we DON'T change your classes)
    weight_before = []
    weight_after = []
    preds_per_step = []
    losses_per_step = []
    dates = []

    # record initial weights (before any update)
    weight_before.append(ensemble.weights.copy())
    dates.append(train50.index[-1])  # last date of train50 as the 'initial' timestamp

    # iterate through the middle 25% online (one obs at a time)
    print("Updating ensemble online over middle 25% and recording weights ...")
    for idx, y_t in middle25.items():
        # compute each member's prediction now (before update)
        preds = np.array([m.predict(step=1)[0] for m in ensemble.models])
        losses = (preds - y_t) ** 2

        preds_per_step.append(preds.copy())
        losses_per_step.append(losses.copy())
        weight_before.append(ensemble.weights.copy())
        dates.append(idx)

        # call original update with a single-element array (so it updates step-by-step)
        ensemble.update(np.array([y_t]))

        # record weights after this update
        weight_after.append(ensemble.weights.copy())

    # For consistency, also append final weight vector into weight_before list (it will be last)
    weight_before.append(ensemble.weights.copy())

    # Forecast final 25% using ensemble (no further updates)
    results['Ensemble'] = ensemble.predict(step=h)

    # --- Compute metrics on final 25% ---
    def rmse(a,b): return np.sqrt(np.mean((a-b)**2))
    def mae(a,b): return np.mean(np.abs(a-b))
    def mape(a,b): return np.mean(np.abs((a-b)/b))*100

    actual = final25.values
    metrics = []
    for name, fc in results.items():
        metrics.append({
            'model': name,
            'RMSE': rmse(actual, fc),
            'MAE': mae(actual, fc),
            'MAPE': mape(actual, fc)
        })
    metrics_df = pd.DataFrame(metrics).set_index('model')
    print("\nPerformance on final 25% (holdout):")
    print(metrics_df.round(4))

    # --- Print weight-summary & quick diagnostics (no files) ---
    # Build DataFrame of weight evolution. Use dates list (initial + update timestamps)
    weight_index = pd.to_datetime(dates)
    weights_mat = np.vstack(weight_before[:len(weight_index)])  # align lengths
    member_names = [type(m).__name__ for m in ensemble.models]
    weights_df = pd.DataFrame(weights_mat, columns=member_names, index=weight_index)
    weights_df.index.name = "date"

    print("\nEnsemble weight evolution (head & tail):")
    with pd.option_context('display.max_rows', 10, 'display.max_columns', None):
        print(weights_df.head())
        print("...")
        print(weights_df.tail())

    print("\nFinal ensemble weights:")
    print(weights_df.iloc[-1].round(4).to_dict())

    print("\nMean weights during update phase:")
    print(weights_df.mean().round(4).to_dict())

    # Optionally show per-step predictions and losses for each model (brief)
    if len(preds_per_step) > 0:
        preds_arr = np.vstack(preds_per_step)
        losses_arr = np.vstack(losses_per_step)
        preds_df = pd.DataFrame(preds_arr, columns=member_names, index=middle25.index)
        losses_df = pd.DataFrame(losses_arr, columns=member_names, index=middle25.index)
        print("\nExample per-step predictions (first 5 rows):")
        print(preds_df.iloc[:5].round(3))
        print("\nExample per-step squared losses (first 5 rows):")
        print(losses_df.iloc[:5].round(3))

    # --- Plots (displayed, not saved) ---
    plt.figure(figsize=(11,6))
    plt.plot(series.index, series.values, label="Actual", linewidth=1.4)
    for name, fc in results.items():
        plt.plot(final25.index, fc, marker='o', label=f"{name} forecast")
    plt.title("Actual vs forecasts (final 25%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Weight evolution plot
    plt.figure(figsize=(10,5))
    for col in weights_df.columns:
        plt.plot(weights_df.index, weights_df[col], marker='o', label=col)
    plt.title("Ensemble weight evolution (initial, then each update step)")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nDone — no files were saved.")
