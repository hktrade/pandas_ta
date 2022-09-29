import pandas as pd
import pandas_ta as ta
import numpy as np
from futu import *
df = pd.DataFrame() # Empty DataFrame
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
symbol = 'HK.00700'

def cal_adx(df, N=14, M=6):
	hd = df['high'].diff().dropna()
	ld = -df['low'].diff().dropna()
	dmp = pd.DataFrame({'dmp':[0]*len(hd)},index=hd.index)
	dmp[(hd>0) & (ld<0)] = hd
	dmp = dmp.rolling(N).sum().dropna()
	dmm = pd.DataFrame({'dmm':[0]*len(ld)},index=ld.index)
	dmm[(hd<0)&(ld>0)] = ld
	dmm = dmm.rolling(N).sum().dropna()
	temp = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift(1)),\
					  abs(df['low']-df['close'].shift(1))],axis=1).dropna()
	tr = temp.max(axis=1).dropna()
	
	s_index = dmm.index & tr.index &dmp.index
	dmp = dmp.loc[s_index]
	dmm = dmm.loc[s_index]
	tr =tr.loc[s_index]
	pdi = 100*dmp['dmp']/tr
	mdi = dmm['dmm']*100/tr
	
	dx = abs(pdi-mdi)/(pdi+mdi)*100
	adx = dx.rolling(M).mean().dropna()
	adx = pd.DataFrame(adx,columns=['adx'])
	
	
	return adx
ret_sub, err_message = quote_ctx.subscribe(symbol, [SubType.K_1M], subscribe_push=False)
if ret_sub == RET_OK:
	ret, df = quote_ctx.get_cur_kline(symbol, 121, SubType.K_1M)
else:
	print(err_message)

ADX = cal_adx(df)

print(ADX.tail(5))

quote_ctx.close()
sys.exit()
