import pandas as pd
import pandas_ta as ta
import numpy as np
from futu import *
df = pd.DataFrame() # Empty DataFrame


def calOBV(df):

	df['VolByHand'] = df['volume']/100

	df['OBV'] =0
	cnt=1
	while cnt<=len(df)-1:
		if(df.iloc[cnt]['close']>df.iloc[cnt-1]['close']):
			df.loc[cnt,'OBV'] = df.loc[cnt-1,'OBV'] + df.loc[cnt,'VolByHand']
		if(df.iloc[cnt]['close']<df.iloc[cnt-1]['close']):
			df.loc[cnt,'OBV'] = df.loc[cnt-1,'OBV'] - df.loc[cnt,'VolByHand']
		cnt=cnt+1
	return df
	
def obv(df):
    i = 0
    OBV = [0]
    while i < df.index[-1]:
        if df.loc[i + 1, 'close'] - df.loc[i, 'close'] > 0:
            OBV.append(df.loc[i + 1, 'volume'])
        if df.loc[i + 1, 'close'] - df.loc[i, 'close'] == 0:
            OBV.append(0)
        if df.loc[i + 1, 'close'] - df.loc[i, 'close'] < 0:
            OBV.append(-df.loc[i + 1, 'volume'])
        i = i + 1
    OBV = pd.Series(OBV,name='OBV')
    df = df.join(OBV)
    return df
	
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
symbol = 'HK.00700'

ret_sub, err_message = quote_ctx.subscribe(symbol, [SubType.K_1M], subscribe_push=False)
if ret_sub == RET_OK:
	ret, df = quote_ctx.get_cur_kline(symbol, 121, SubType.K_1M)
else:
	print(err_message)

print(obv(df)[['close','OBV']])
print('----------------------')
obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
print(obv)
print('----------------------')
print(df.ta.obv().tail(5))
print('----------------------')
print(calOBV(df)[['close','VolByHand','OBV']])
print('----------------------')

quote_ctx.close()
sys.exit()
