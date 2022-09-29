import pandas as pd
import pandas_ta as ta
import numpy as np
from futu import *
df = pd.DataFrame() # Empty DataFrame
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
symbol = 'HK.00700'

def get_MTM(df,M=6,N=6):
	df['mtm']=df['close']-df['close'].shift(M)
	df['mtmma']=df['mtm'].rolling(N).mean()
	return df


ret_sub, err_message = quote_ctx.subscribe(symbol, [SubType.K_1M], subscribe_push=False)
if ret_sub == RET_OK:
	ret, df = quote_ctx.get_cur_kline(symbol, 121, SubType.K_1M)
else:
	print(err_message)

# key MOM_10
print(df.ta.mom().tail(10))
df_mtm = get_MTM(df)[['mtm','mtmma']]
print(df_mtm.tail(10))
quote_ctx.close()
sys.exit()
