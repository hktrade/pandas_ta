import pandas as pd
import pandas_ta as ta
import numpy as np
from futu import *
df = pd.DataFrame() # Empty DataFrame
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
symbol = 'HK.00700'

def get_TRIX(df,N=12,M=20):
    for i in range(len(df)):
        if i==0:
            df.loc[i,'ema']=df.loc[i,'close']
        if i>0:
            df.loc[i,'ema']=(2*df.loc[i-1,'close']+(N-1)*df.loc[i,'close'])/(N+1)
    for i in range(len(df)):
        if i==0:
            df.loc[i,'ema1']=df.loc[i,'ema']
        if i>0:
            df.loc[i,'ema1']=(2*df.loc[i-1,'ema']+(N-1)*df.loc[i,'ema'])/(N+1)
    for i in range(len(df)):
        if i==0:
            df.loc[i,'tr']=df.loc[i,'ema1']
        if i>0:
            df.loc[i,'tr']=(2*df.loc[i-1,'ema1']+(N-1)*df.loc[i,'ema1'])/(N+1)
    df['trix']=100*(df['tr']-df['tr'].shift(1))/df['tr'].shift(1)
    df['trma']=df['trix'].rolling(M).mean()
    return df

ret_sub, err_message = quote_ctx.subscribe(symbol, [SubType.K_1M], subscribe_push=False)
if ret_sub == RET_OK:
	ret, df = quote_ctx.get_cur_kline(symbol, 121, SubType.K_1M)
else:
	print(err_message)


print(df.ta.trix().tail(5))

df_t = get_TRIX(df);

print(df_t[['close', 'tr', 'trix','trma']].tail(5))

quote_ctx.close()
sys.exit()
