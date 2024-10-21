# Monte Carlo Option Pricer

This project implements a Monte Carlo simulation to price European and American options. 
It uses geometric Brownian motion, variance reduction techniques, and parallel processing.

## Features
- Simulates asset prices using geometric Brownian motion
- Prices European options with a payoff at maturity
- Prices American options with early exercise using Least Squares Monte Carlo (LSMC)
- Uses variance reduction (antithetic variates)
- Supports parallel processing for efficiency
