import pymc3 as pm
import yfinance as yf
import numpy as np
import torch
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# Download historical data for desired ticker symbol
data = yf.download('AAPL', start='2020-01-01', end='2022-12-31')

def bayesian():
    returns = data['Close'].pct_change()

    # Drop the first element which is NaN
    returns = returns.dropna()*100

    with pm.Model() as model:
        # Define priors
        sigma = pm.HalfNormal('sigma', sd=1)
        mu = pm.Normal('mu', 0, sd=1)

        # Define likelihood
        likelihood = pm.Normal('y', mu=mu, sd=sigma, observed=returns)

        # Inference
        start = pm.find_MAP() # Use MAP estimate (optimization) as the initial state for MCMC
        step = pm.NUTS(scaling=start) # Use the No-U-Turn Sampler
        trace = pm.sample(2000, step, start=start, progressbar=True) # Draw 2000 posterior samples using NUTS sampling

    pm.traceplot(trace)

    # Save to file
    pm.backends.text.dump('trace', trace)

    # Print as table
    print(pm.summary(trace))
    pm.summary(trace).to_csv('summary.csv')

def train():
    # Generate some example data
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1))

    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Initialize weights
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    # Set hyperparameters
    alpha = 1.0
    lambda_ = 1.0

    # Training loop
    for _ in range(1000):
        # Compute predictions
        y_pred = w * X + b

        # Compute loss
        loss = torch.mean((y - y_pred)**2) + alpha * w**2 + lambda_ * b**2

        # Backpropagation
        loss.backward()

        # Update weights
        with torch.no_grad():
            w -= 0.01 * w.grad
            b -= 0.01 * b.grad

        # Reset gradients
        w.grad.zero_()
        b.grad.zero_()

    print(w, b)

if __name__ == '__main__':
    train()