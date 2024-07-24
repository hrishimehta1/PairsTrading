import pandas as pd
import numpy as np
import yfinance as yf
import csv
import os
import pendulum
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import datetime
import time

cwd = os.getcwd()

# Get Ticker Data for any Ticker passed in
def get_ticker_data(ticker):
    try:
        ticker_data = yf.Ticker(ticker).history(period='1y', interval='60m', actions=False)
        return ticker_data
    except Exception as e:
        print("Error fetching data for:", ticker, e)
        with open('error_log.txt', 'a') as f:
            f.write(f'Error fetching data for {ticker}: {e}')
            f.write('\n')
        return None

# Average price for entire price series of two tickers
def calculate_average_price(ticker_a_closing_prices, ticker_b_closing_prices):
    a_avg_price = sum(ticker_a_closing_prices) / len(ticker_a_closing_prices)
    b_avg_price = sum(ticker_b_closing_prices) / len(ticker_b_closing_prices)
    return [a_avg_price, b_avg_price]

# Get Pair Data and assign Stock A and B ensuring A is the lower priced stock so ratio is > 1
def get_pair(ticker_a, ticker_b):
    ticker_a_data = get_ticker_data(ticker_a)
    ticker_b_data = get_ticker_data(ticker_b)

    if ticker_a_data is None or ticker_b_data is None:
        return None

    ticker_a_closing_prices = list(ticker_a_data['Close'])
    ticker_b_closing_prices = list(ticker_b_data['Close'])

    if len(ticker_a_closing_prices) != len(ticker_b_closing_prices):
        print("Error: Length of arrays are not the same for", ticker_a, ticker_b)
        with open('error_log.txt', 'a') as f:
            f.write(f'Error: Length of arrays are not the same for {ticker_a} {ticker_b}')
            f.write('\n')
        return None

    [a_avg_price, b_avg_price] = calculate_average_price(ticker_a_closing_prices, ticker_b_closing_prices)

    if a_avg_price > b_avg_price:
        return [[ticker_b, ticker_b_data, ticker_b_closing_prices], [ticker_a, ticker_a_data, ticker_a_closing_prices]]
    else:
        return [[ticker_a, ticker_a_data, ticker_a_closing_prices], [ticker_b, ticker_b_data, ticker_b_closing_prices]]

# Calculation Logic 
def calculate_residual_spread(stock_1_closing_prices, stock_2_closing_prices, gamma, market_excess_return):
    residual_spread = []
    for i in range(len(stock_1_closing_prices)):
        residual = (stock_1_closing_prices[i] - stock_2_closing_prices[i]) - gamma * market_excess_return[i]
        residual_spread.append(residual)
    return residual_spread

def vasicek_mean_reversion(x, theta, kappa, sigma, dt):
    return theta + (x - theta) * np.exp(-kappa * dt) + sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))

def kalman_filter(y, delta_t, theta, kappa, sigma):
    F = np.array([[np.exp(-kappa * delta_t)]])
    H = np.array([[1]])
    Q = np.array([[sigma**2 * (1 - np.exp(-2 * kappa * delta_t)) / (2 * kappa)]])
    R = np.array([[1]])  # Measurement noise covariance

    initial_state_mean = theta
    initial_state_covariance = np.array([[sigma**2 / (2 * kappa)]])

    kf = KalmanFilter(transition_matrices=F, observation_matrices=H,
                      transition_covariance=Q, observation_covariance=R,
                      initial_state_mean=initial_state_mean,
                      initial_state_covariance=initial_state_covariance)
    return kf.smooth(y)

def trading_logic(stock_1, stock_2, stock_1_closing_prices, stock_2_closing_prices, market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold):
    residual_spread = calculate_residual_spread(stock_1_closing_prices, stock_2_closing_prices, gamma, market_excess_return)
    smoothed_state_means, _ = kalman_filter(residual_spread, delta_t, theta, kappa, sigma)

    open_positions = []
    closed_positions = []
    position_open = False

    for i in range(len(smoothed_state_means)):
        if not position_open and abs(smoothed_state_means[i] - theta) > threshold:
            open_positions.append(i)
            position_open = True
        elif position_open and abs(smoothed_state_means[i] - theta) < convergence_threshold:
            closed_positions.append(i)
            position_open = False

    return open_positions, closed_positions

def create_pair_dataframe(stock_1, stock_2, stock_1_closing_prices, stock_2_closing_prices, pair_daily_ratios, pair_daily_spreads):
    df = pd.DataFrame({stock_1 : stock_1_closing_prices,
                       stock_2 : stock_2_closing_prices,
                       'Ratio' : pair_daily_ratios,
                       'Spread' : pair_daily_spreads})
    if not os.path.exists(cwd + '/pairs'):
        os.makedirs(cwd + '/pairs')
    df.to_csv(cwd + '/pairs/' + f'{stock_1}_{stock_2}.csv')

    return df

# Creating a Graph / Chart of Spread
def create_graph(stock_1, stock_2, stock_1_data, pair_daily_spreads, st_dev, r_sq):
    dt_list = [pendulum.parse(str(dt)).float_timestamp for dt in list(stock_1_data.index)]
    plt.style.use('dark_background')
    plt.plot(dt_list, pair_daily_spreads, linewidth=2)
    plt.axhline(y=st_dev*2, xmin=0.0, xmax=1.0, color='m')
    plt.axhline(y=st_dev, xmin=0.0, xmax=1.0, color='r')
    plt.axhline(y=0, xmin=0.0, xmax=1.0, color='w')
    plt.axhline(y=(st_dev*-1), xmin=0.0, xmax=1.0, color='r')
    plt.axhline(y=(st_dev*-2), xmin=0.0, xmax=1.0, color='m')
    if r_sq > .65:
        plt.savefig((cwd + '/figs_high/' + f'{stock_1}_{stock_2}.png'))
        plt.clf()
    else:
        plt.savefig((cwd + '/figs_low/' + f'{stock_1}_{stock_2}.png'))
        plt.clf()

def backtest_pair(stock_1, stock_2, stock_1_closing_prices, stock_2_closing_prices, pair_daily_spreads, st_dev, r_sq, average_ratio):
    trades_pnl = []
    trade_enter = []
    trade_exit = []
    hold_period = []
    max_open_loss = 0
    open_price = 0
    close_price = 0
    for x in range(0, len(pair_daily_spreads)):
        if pair_daily_spreads[x] > st_dev and open_price == 0:
            open_price = pair_daily_spreads[x]
            trade_enter.append(x)
        elif pair_daily_spreads[x] < st_dev * -1 and open_price == 0:
            open_price = pair_daily_spreads[x]
            trade_enter.append(x)
        if open_price > 1 and pair_daily_spreads[x] < 1:
            close_price = pair_daily_spreads[x]
            trades_pnl.append(open_price - close_price)
            trade_exit.append(x)
            hold_period.append(trade_exit[len(trade_exit) - 1] - trade_enter[len(trade_enter) - 1])
            open_price = 0
            close_price = 0
        if open_price < -1 and pair_daily_spreads[x] > -1:
            close_price = pair_daily_spreads[x]
            trades_pnl.append(close_price - open_price)
            trade_exit.append(x)
            hold_period.append(trade_exit[len(trade_exit) - 1] - trade_enter[len(trade_enter) - 1])
            open_price = 0
            close_price = 0
        if open_price > 1:
            open_pnl = open_price - pair_daily_spreads[x]
            if open_pnl < max_open_loss:
                max_open_loss = open_pnl
        if open_price < -1:
            open_pnl = pair_daily_spreads[x] - open_price
            if open_pnl < max_open_loss:
                max_open_loss = open_pnl

    capital_leg_1 = stock_1_closing_prices[0] * average_ratio * 100
    capital_leg_2 = stock_2_closing_prices[0] * 100

    total_capital = capital_leg_1 + capital_leg_2
    total_return = (sum(trades_pnl) * 100) / total_capital

    print("trades PnL: ", trades_pnl)
    print("trade enter: ", trade_enter)
    print("trade exit: ", trade_exit)
        print("hold period: ", hold_period)
    print("average hold period: ", sum(hold_period) / len(hold_period))
    print("largest winner: ", max(trades_pnl) / (total_capital / 100))
    print("max open loss: ", max_open_loss / (total_capital / 100))
    print("number of round trips: ", len(hold_period))
    print("total PnL per unit", sum(trades_pnl))
    print("total profit per 100 shares: $", sum(trades_pnl) * 100)
    print("capital used on leg 1:", stock_1_closing_prices[0] * average_ratio * 100)
    print("capital used on leg 2:", stock_2_closing_prices[0] * 100)
    print("total capital used: ", total_capital)
    print("total return: ", total_return * 100, "%")

    if r_sq > .65:
        with open('candidates_high.csv', 'a', newline='') as csvfile:
            fieldnames = ['Stock 1', 'Stock 2', 'Average Ratio', 'Total Capital Used', 'Standard Dev', 'Average Hold', 'Trade Count', 'Largest Winner', 'Max Open Loss', 'R Squared', 'Total Return']
            thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            thewriter.writerow({'Stock 1': stock_1,
                                'Stock 2': stock_2,
                                'Average Ratio': average_ratio,
                                'Total Capital Used': total_capital,
                                'Standard Dev': st_dev,
                                'Average Hold': sum(hold_period) / len(hold_period),
                                'Trade Count': len(hold_period),
                                'Largest Winner': max(trades_pnl) / (total_capital / 100),
                                'Max Open Loss': max_open_loss / (total_capital / 100),
                                'R Squared': r_sq,
                                'Total Return': total_return})
    else:
        with open('candidates_low.csv', 'a', newline='') as csvfile:
            fieldnames = ['Stock 1', 'Stock 2', 'Average Ratio', 'Total Capital Used', 'Standard Dev', 'Average Hold', 'Trade Count', 'Largest Winner', 'Max Open Loss', 'R Squared', 'Total Return']
            thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            thewriter.writerow({'Stock 1': stock_1,
                                'Stock 2': stock_2,
                                'Average Ratio': average_ratio,
                                'Total Capital Used': total_capital,
                                'Standard Dev': st_dev,
                                'Average Hold': sum(hold_period) / len(hold_period),
                                'Trade Count': len(hold_period),
                                'Largest Winner': max(trades_pnl) / (total_capital / 100),
                                'Max Open Loss': max_open_loss / (total_capital / 100),
                                'R Squared': r_sq,
                                'Total Return': total_return})

# Pairs Trading Main

def pairs_trade(ticker_a, ticker_b, market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold): 
    try:
        pair = get_pair(ticker_a, ticker_b)
        if pair is None:
            return
        [[stock_1, stock_1_data, stock_1_closing_prices], [stock_2, stock_2_data, stock_2_closing_prices]] = pair
    except Exception as e:
        print("Error:", e)
        return

    pair_daily_ratios = calculate_ratio(stock_1_closing_prices, stock_2_closing_prices)
    average_ratio = sum(pair_daily_ratios) / len(pair_daily_ratios)
    
    pair_daily_spreads = calculate_spreads(stock_1_closing_prices, stock_2_closing_prices, pair_daily_ratios, average_ratio)
    st_dev = statistics.stdev(pair_daily_spreads)
    r_sq = calculate_r_squared_correlation(stock_1, stock_2, stock_1_closing_prices, stock_2_closing_prices)
    
    create_pair_dataframe(stock_1, stock_2, stock_1_closing_prices, stock_2_closing_prices, pair_daily_ratios, pair_daily_spreads)
    
    try:
        open_positions, closed_positions = trading_logic(stock_1, stock_2, stock_1_closing_prices, stock_2_closing_prices, market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold)
        backtest_pair(stock_1, stock_2, stock_1_closing_prices, stock_2_closing_prices, pair_daily_spreads, st_dev, r_sq, average_ratio)
    except Exception as e:
        print("Error backtesting trade performance for", stock_1, "and", stock_2, ":", e)
        with open('error_log.txt', 'a') as f:
            f.write(f'Error backtesting trade performance for {stock_1} and {stock_2}: {e}')
            f.write('\n')

def sector(stock_list, market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold):
    for x in range(len(stock_list)):
        for y in range(len(stock_list)):
            if x != y:
                print("Pair: ", stock_list[x], stock_list[y])
                pairs_trade(stock_list[x], stock_list[y], market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold)

def run_pairs_all_sectors(market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold):
    sector(xly, market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold)
    sector(xlp, market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold)
    sector(xlf, market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold)
    sector(xlv, market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold)
    sector(xli, market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold)
    sector(xlb, market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold)
    sector(xlk, market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold)
    sector(qqq, market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold)
    sector(xlc, market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold)
    sector(smh, market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold)

nowtime = str(datetime.datetime.now())


market_excess_return = np.random.normal(0, 0.01, 365)

# Parameters for the Vasicek model and trading logic
theta = 0
kappa = 0.1
sigma = 0.1
gamma = 0.5
delta_t = 1 / 252
threshold = 2  # Example threshold for opening positions
convergence_threshold = 0.5  # Example threshold for closing positions

# Run once on first launch, then scheduler takes over
run_pairs_all_sectors(market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold)
print("Initial run completed at:", nowtime)

def job(t):
    try:
        os.remove('error_log.txt')
        os.remove('candidates_high.csv')
        os.remove('candidates_low.csv')
    except Exception as e:
        print("Error cleaning up old logs:", e)
    time.sleep(5)
    run_pairs_all_sectors(market_excess_return, theta, kappa, sigma, gamma, delta_t, threshold, convergence_threshold)
    print("Scheduler run completed at:", str(datetime.datetime.now()), t)

for i in ["06:40", "07:15", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00"]:
    schedule.every().monday.at(i).do(job, i)
    schedule.every().tuesday.at(i).do(job, i)
    schedule.every().wednesday.at(i).do(job, i)
    schedule.every().thursday.at(i).do(job, i)
    schedule.every().friday.at(i).do(job, i)

while True:
    schedule.run_pending()
    time.sleep(30)

