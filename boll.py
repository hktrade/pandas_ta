import pandas as pd
import pandas_ta as ta
from futu import *
df = pd.DataFrame() # Empty DataFrame

quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
symbol = 'HK.800000'

ret_sub, err_message = quote_ctx.subscribe(symbol, [SubType.K_1M], subscribe_push=False)
if ret_sub == RET_OK:
	ret, df = quote_ctx.get_cur_kline(symbol, 121, SubType.K_1M)
else:
	print(err_message)

df.set_index(pd.DatetimeIndex(df["time_key"]), inplace=True)

df.ta.log_return(cumulative=True, append=True)
df.ta.percent_return(cumulative=True, append=True)

df.columns

print(df.ta.bbands)

bb = df.ta.bbands(length=20)
print(bb[['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']],bb.keys())
quote_ctx.close()
sys.exit()
