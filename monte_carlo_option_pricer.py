import numpy as np
from scipy.stats import norm
import multiprocessing as mp

# Geometric Brownian Motion to simulate asset prices
def simulate_gbm(S0, r, sigma, T, N, M, antithetic=True):
    """
    Simulate asset price paths using geometric Brownian motion.
    
    Parameters:
    S0 : Initial stock price
    r  : Risk-free rate
    sigma : Volatility
    T  : Time to maturity
    N  : Number of time steps
    M  : Number of simulations
    antithetic : Use antithetic variates (default True)
    
    Returns:
    A matrix of simulated asset prices with M paths and N+1 time steps.
    """
    dt = T / N
    Z = np.random.normal(0, 1, (M, N))
    
    if antithetic:
        Z = np.concatenate((Z, -Z), axis=0)
    
    S = np.zeros((Z.shape[0], N+1))
    S[:, 0] = S0
    for t in range(1, N+1):
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
    
    return S

# European Option Payoff
def european_option_payoff(S, K, option_type='call'):
    """
    Calculate the payoff for European options.
    
    Parameters:
    S : Simulated asset prices at maturity (last column of the GBM simulation)
    K : Strike price
    option_type : 'call' or 'put'
    
    Returns:
    Payoff values for all paths.
    """
    if option_type == 'call':
        return np.maximum(S - K, 0)
    elif option_type == 'put':
        return np.maximum(K - S, 0)
    else:
        raise ValueError("Option type must be 'call' or 'put'.")

# American Option Payoff and Pricing
def american_option_pricer(S, K, r, T, N, option_type='call'):
    """
    Price an American option using Monte Carlo and Least Squares.
    
    Parameters:
    S : Simulated asset prices
    K : Strike price
    r : Risk-free rate
    T : Time to maturity
    N : Number of time steps
    option_type : 'call' or 'put'
    
    Returns:
    The price of the American option.
    """
    dt = T / N
    payoff = np.zeros_like(S)
    
    if option_type == 'call':
        payoff[:, -1] = np.maximum(S[:, -1] - K, 0)
    elif option_type == 'put':
        payoff[:, -1] = np.maximum(K - S[:, -1], 0)
    else:
        raise ValueError("Option type must be 'call' or 'put'.")

    for t in range(N-1, 0, -1):
        itm_paths = np.where(S[:, t] > K if option_type == 'call' else S[:, t] < K)[0]
        regression = np.polyfit(S[itm_paths, t], payoff[itm_paths, t+1] * np.exp(-r*dt), 2)
        continuation_value = np.polyval(regression, S[itm_paths, t])
        exercise_value = np.maximum(S[itm_paths, t] - K if option_type == 'call' else K - S[itm_paths, t], 0)
        
        payoff[itm_paths, t] = np.where(exercise_value > continuation_value, exercise_value, payoff[itm_paths, t+1] * np.exp(-r*dt))
    
    return np.mean(payoff[:, 1] * np.exp(-r*dt))

# Black-Scholes formula for European option validation
def black_scholes_price(S0, K, r, sigma, T, option_type='call'):
    """
    Calculate European option price using the Black-Scholes formula.
    
    Parameters:
    S0 : Initial stock price
    K : Strike price
    r : Risk-free rate
    sigma : Volatility
    T : Time to maturity
    option_type : 'call' or 'put'
    
    Returns:
    The Black-Scholes price of the European option.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'.")

# Monte Carlo pricing function
def monte_carlo_pricer(S0, K, r, sigma, T, N, M, option_type='call', option_style='european'):
    """
    Price an option using Monte Carlo simulation with parallel processing.
    
    Parameters:
    S0 : Initial stock price
    K : Strike price
    r : Risk-free rate
    sigma : Volatility
    T : Time to maturity
    N : Number of time steps
    M : Number of simulations
    option_type : 'call' or 'put'
    option_style : 'european' or 'american'
    
    Returns:
    Estimated option price.
    """
    num_cores = mp.cpu_count()
    pool = mp.Pool(num_cores)

    # Split simulations for parallel processing
    simulations_per_core = M // num_cores
    results = []
    
    for _ in range(num_cores):
        if option_style == 'european':
            result = pool.apply_async(european_monte_carlo_simulation, args=(S0, K, r, sigma, T, N, simulations_per_core, option_type))
        elif option_style == 'american':
            result = pool.apply_async(american_monte_carlo_simulation, args=(S0, K, r, sigma, T, N, simulations_per_core, option_type))
        results.append(result)

    pool.close()
    pool.join()

    # Aggregate the results
    option_prices = [result.get() for result in results]
    
    return np.mean(option_prices)

# Helper for European option simulation
def european_monte_carlo_simulation(S0, K, r, sigma, T, N, M, option_type):
    S = simulate_gbm(S0, r, sigma, T, N, M)
    payoff = european_option_payoff(S[:, -1], K, option_type)
    return np.mean(payoff) * np.exp(-r * T)

# Helper for American option simulation
def american_monte_carlo_simulation(S0, K, r, sigma, T, N, M, option_type):
    S = simulate_gbm(S0, r, sigma, T, N, M)
    return american_option_pricer(S, K, r, T, N, option_type)

# Example usage
if __name__ == "__main__":
    # Option parameters
    S0 = 100  # Initial stock price
    K = 100   # Strike price
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    T = 1.0  # Time to maturity (1 year)
    N = 100  # Number of time steps
    M = 10000  # Number of simulations

    # European option price using Monte Carlo
    european_price = monte_carlo_pricer(S0, K, r, sigma, T, N, M, option_type='call', option_style='european')
    print(f"Monte Carlo European Call Price: {european_price}")

    # American option price using Monte Carlo
    american_price = monte_carlo_pricer(S0, K, r, sigma, T, N, M, option_type='call', option_style='american')
    print(f"Monte Carlo American Call Price: {american_price}")

    # Black-Scholes price for validation
    bs_price = black_scholes_price(S0, K, r, sigma, T, option_type='call')
    print(f"Black-Scholes Call Price: {bs_price}")
