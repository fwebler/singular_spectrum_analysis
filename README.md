### README:

---

## Singular Spectrum Analysis (SSA) for Time Series

This Python class provides an implementation of the Singular Spectrum Analysis (SSA) for time series gap filling and forecasting.

### Features:
- Time series decomposition
- Gap filling in time series data
- Forecasting future values

### Usage:

1. **Initialization**:
    Initialize the SSA object with your time series data.
    ``` python
    ssa = SSA(your_time_series)
    ```

2. **Embedding**:
    Embed the time series data.
    ``` python
    ssa.embed()
    ```

3. **Decomposition**:
    Decompose the time series into its singular values.
    ``` python
    ssa.decompose()
    ```

4. **Gap Filling**:
    If your time series has missing values, you can fill them using:
    ``` python
    ssa.forecast_recurrent(steps_ahead=0)
    ```

5. **Forecasting**:
    Predict future values of your time series by specifying the number of steps ahead you want to forecast.
    ``` python
    ssa.forecast_recurrent(steps_ahead=number_of_steps)
    ```
