import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta

def build_dashboard():
    # --- Configuration ---
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY"] # SPY as benchmark
    benchmark = "SPY"
    lookback_days = 365 * 2
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    print(f"Fetching data for {tickers} from {start_date.date()} to {end_date.date()}...")
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Separate benchmark from assets
    assets = data.drop(columns=[benchmark])
    bench_prices = data[benchmark]
    
    # 1. Returns (Daily)
    returns = assets.pct_change().dropna()
    bench_returns = bench_prices.pct_change().dropna()
    
    # Portfolio: Equal Weighted
    weights = np.array([1/len(assets.columns)] * len(assets.columns))
    portfolio_returns = returns.dot(weights)
    
    # Cumulative Returns
    cum_portfolio = (1 + portfolio_returns).cumprod() - 1
    cum_bench = (1 + bench_returns).cumprod() - 1
    
    # 2. Volatility (Annualized Rolling 30-day)
    # Vol = std(daily_returns) * sqrt(252)
    rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(252)
    
    # 3. Predictions (Monte Carlo Simulation)
    # We predict the portfolio value based on historical drift and volatility
    last_price = 1.0 # Start at 1 (normalized)
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    
    num_simulations = 100
    num_days = 252 # Predict next year
    
    simulations = np.zeros((num_days, num_simulations))
    for i in range(num_simulations):
        # Geometric Brownian Motion: S_t = S_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        daily_returns_sim = np.random.normal(mu, sigma, num_days)
        simulations[:, i] = last_price * np.exp(np.cumsum(daily_returns_sim))

    # --- Visualization (The Dashboard) ---
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2)
    
    # Hero Chart: Cumulative Returns
    ax_hero = fig.add_subplot(gs[0, :])
    ax_hero.plot(cum_portfolio, label="Equal-Weighted Portfolio", color="#2ecc71", lw=2.5)
    ax_hero.plot(cum_bench, label=f"Benchmark ({benchmark})", color="#e74c3c", lw=1.5, ls='--')
    ax_hero.set_title("Portfolio Performance vs Benchmark", fontsize=16, fontweight="bold")
    ax_hero.set_ylabel("Cumulative Return")
    ax_hero.legend()
    
    # Volatility Chart
    ax_vol = fig.add_subplot(gs[1, 0])
    ax_vol.plot(rolling_vol, color="#3498db", lw=1.5)
    ax_vol.set_title("Portfolio Annualized Volatility (30d Rolling)", fontsize=14)
    ax_vol.set_ylabel("Volatility")
    
    # Correlation Heatmap (Seaborn)
    ax_corr = fig.add_subplot(gs[1, 1])
    corr_matrix = returns.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="RdYlGn", center=0, ax=ax_corr, fmt=".2f")
    ax_corr.set_title("Asset Correlations", fontsize=14)
    
    # Prediction Chart (Monte Carlo Fan)
    ax_pred = fig.add_subplot(gs[2, :])
    for i in range(num_simulations):
        ax_pred.plot(simulations[:, i], color="gray", alpha=0.1, lw=0.5)
    
    # Plot the mean prediction
    ax_pred.plot(np.mean(simulations, axis=1), color="#f1c40f", lw=3, label="Expected Path")
    ax_pred.set_title("Forward Projection: Monte Carlo Simulation (Next 252 Days)", fontsize=14)
    ax_pred.set_ylabel("Normalized Value")
    ax_pred.set_xlabel("Days")
    ax_pred.legend()
    
    plt.tight_layout()
    output_file = "portfolio_dashboard.png"
    plt.savefig(output_file)
    print(f"Dashboard saved to {output_file}")


if __name__ == "__main__":
    build_dashboard()
