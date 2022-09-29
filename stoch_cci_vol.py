import pandas as pd
import pandas_ta as ta
import numpy as np
from futu import *
df = pd.DataFrame() # Empty DataFrame
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
symbol = 'HK.00700'

# def StochRSI(cls, close, m,  p):
  # RSI = talib.RSI(np.array(close), timeperiod=m)  
  # LLV= RSI .rolling(window=m).min()
  # HHV= RSI .rolling(window=m).max()
  # stochRSI = (RSI  - LLV) / (HHV - LLV) * 100
  # fastk = talib.MA(np.array(stochRSI)  , p)
  # fastd = talib.MA(np.array(fastk), p)
  # return fastk , fastd 
  
ret_sub, err_message = quote_ctx.subscribe(symbol, [SubType.K_1M], subscribe_push=False)
if ret_sub == RET_OK:
	ret, df = quote_ctx.get_cur_kline(symbol, 121, SubType.K_1M)
else:
	print(err_message)

print(df.ta.pvol().tail(5))

print(df.ta.stoch().tail(5))

print('\n')

print(df.ta.cci().tail(5))

quote_ctx.close()
sys.exit()
