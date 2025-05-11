# Define the buy signal logic
def buy_shares(cash, actual_price_today, fraction=1.0):
    """
    Buys as many shares as possible with the fraction of available cash.
    Returns the updated cash and number of shares bought.
    """
    shares = int(cash * fraction / actual_price_today)
    cash -= shares * actual_price_today
    return cash, shares

# Define the sell signal logic
def sell_shares(cash, shares, actual_price_today, fraction=1.0):
    """
    Sells a fraction of the shares owned.
    Returns the updated cash and sets shares to new value.
    """
    shares_to_sell = int(shares * fraction)
    cash += shares_to_sell * actual_price_today
    shares -= shares_to_sell
    return cash, shares

def simulate_trading_strategy(
    actual: np.ndarray,
    predicted: np.ndarray,
    action: callable,
    initial_cash: float = 10000.0
) -> tuple:
    """
    Simulates a trading strategy based on the actual and predicted stock prices.
    The action function should define the buy/sell logic. Which takes the current
    cash, actual price, and predicted price. The function returns a tuple 
    containing the updated cash and number of shares and portfolio value over 
    each iteration. The action function should return a positive value to buy,
    negative value to sell, and zero to do nothing. These abs(result) of the action
    function will be passed to the buy/sell function as fraction of stock to
    sell or fraction of cash used to buy.
    """
    # Initialize variables
    cash = initial_cash
    shares = [0]
    portfolio_value = [initial_cash]
    
    # Iterate through the actual and predicted prices
    for actual_price_today, predicted_price_today in zip(actual, predicted):
        # Buy shares if the predicted price is higher than the actual price
        action_value = action(cash, actual_price_today, predicted_price_today)
        
        if action_value > 0: # Buy shares
            cash, shares_today = buy_shares(cash, actual_price_today, abs(action_value))
            shares.append(shares_today)
        elif action_value < 0: # Sell shares
            cash, shares_today = sell_shares(cash, shares[-1], actual_price_today, abs(action_value))
            shares.append(shares_today)
        else: # No action
            shares.append(shares[-1])
            
        # Calculate the the portfolio value after each iteration
        portfolio_value.append(cash + shares[-1] * actual_price_today)
    
    return cash, shares, portfolio_value

# This uses a random action function to decide whether to buy or sell shares
def random_action(*args, **kwargs):
    """
    Randomly decides to buy or sell shares.
    """
    return np.random.choice([.5, 1, -.5])

# Use the predicted stock to determine value to determine the action.
def prediction_action(actual, predicted):
    """
    Decides to buy or sell shares based on the predicted price.
    """
    if predicted > actual:
        return .5 # Buy with half of the cash
    elif actual > predicted:
        return -.5 # Sell half of the shares
    else:
        return 0 # No action
    
    
    # Random strategy simulation store
random_results = {}

# Simulate the trading strategy for each stock.
for ticker, (dataframe, X_test, y_test, scaler, results) in test_data.items():
    
    cash, shares, portfolio_value = simulate_trading_strategy(
        np.array(results['Actual'].values),
        np.array(results['Prediction'].values),
        random_action,
        initial_cash=10000.0
    )
    
    # Create a DataFrame to hold the results
    data = pd.DataFrame({
        'Date': results['Date'],
        'Actual': results['Actual'],
        'Prediction': results['Prediction'],
        'Shares': shares[1:],
        'Portfolio Value': portfolio_value[1:],
    })
    
    # Add the final cash value to the DataFrame
    random_results[ticker] = data