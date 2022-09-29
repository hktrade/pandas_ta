from futu import *
import pandas as pd
import math

quote_ctx = OpenQuoteContext(host='118.190.162.202', port=11111)
symbol = 'HK.00700'
ret_sub, err_message = quote_ctx.subscribe(symbol, [SubType.K_3M], subscribe_push=False)
if ret_sub == RET_OK:
	ret, df = quote_ctx.get_cur_kline(symbol, 111, SubType.K_3M)

df.set_index(['time_key'], inplace=True)
df.index = pd.to_datetime(df.index)
df.index.name = 'time_key'

df['VWAP'] = df.turnover.cumsum() / df.volume.cumsum()

print(df['VWAP'])

quote_ctx.close()
sys.exit()