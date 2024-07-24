# Pairs Trading Strategy with Vasicek Mean Reversion Model and Kalman Filter

This repository contains a Python implementation of a pairs trading strategy using a combination of the Vasicek mean reversion model and Kalman filter. The strategy is based on identifying and exploiting price inefficiencies between pairs of correlated stocks.

## Components and Rationale

### 1. Residual Spread Calculation
- **Equation**: \[
  \text{residual} = (P_{\text{stock1}} - P_{\text{stock2}}) - \gamma \times \text{market\_excess\_return}
  \]
- **Rationale**: Adjusting for market excess returns isolates stock-specific price movements, removing broad market trends. This focuses the model on the unique relationship between the stock pair, filtering out systematic risk and honing in on idiosyncratic risk.

### 2. Vasicek Mean Reversion Model
- **Equation**: \[
  dx_t = \kappa (\theta - x_t) \, dt + \sigma \, dB_t
  \]
- **Rationale**: The Vasicek model is a classic stochastic process used to describe mean-reverting behavior. It captures the tendency of the spread to revert to its historical mean. This provides a robust mathematical framework for predicting price convergence, which is fundamental in pairs trading.

### 3. Kalman Filter
- **Equations**: 
  \[
  x_t = F x_{t-1} + w_t
  \]

  \[
  y_t = H x_t + v_t
  \]
- **Rationale**: The Kalman filter estimates and smooths the residual spread, providing clearer trading signals by filtering out noise. This enhances the accuracy of trading signals, making the model more robust in real-time trading.

## Advantages of the Model
1. **Captures Mean Reversion**: The model accurately captures the mean reversion property underlying pairs trading. Using logarithmic differences of prices ensures the spread remains stable even as prices fluctuate.
2. **Continuous Time Model**: Convenient for forecasting purposes, allowing traders to compute expected convergence times and answer critical questions regarding expected holding periods and returns.
3. **Tractability**: The model is easily estimated using the Kalman filter in a state space setting. The maximum likelihood estimator used is optimal in terms of minimum mean square error (MMSE).

## Trading Rules
The trading strategy opens positions based on the accumulated residual spread and unwinds them when the accumulated spread converges to the long-term mean. Specifically:
- **Open Positions**: When the accumulated spread exceeds a certain threshold.
- **Close Positions**: When the spread reverts to within a convergence threshold.

## Implementation

### Data Retrieval
Fetch historical price data using the `yfinance` API.

### Pairs Selection
Choose stock pairs based on their average prices to ensure meaningful ratio calculations.

### Calculation Logic
Calculate returns, residual spreads, and apply the Vasicek mean reversion model and Kalman filter to generate trading signals.

### Backtesting
Simulate trades using historical data to evaluate the strategy's effectiveness. Calculate key metrics such as profit and loss (PnL), Sharpe ratio, and maximum drawdown.
