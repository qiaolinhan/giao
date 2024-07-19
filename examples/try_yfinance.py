import yfinance as yf

# get all info
nvidia = yf.Ticker("NVDA")

# get meta information
meta1 = nvidia.history_metadata

print("======>[INFO] meta info of NVDA:", meta1)
