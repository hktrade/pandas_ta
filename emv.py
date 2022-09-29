import pandas as pd
import pandas_ta as ta
import numpy as np
from futu import *
from idc import *
df = pd.DataFrame() # Empty DataFrame
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
symbol = 'HK.00700'


def EMV(HIGH,LOW,VOL,N=14,M=9):                     #简易波动指标 
    VOLUME=MA(VOL,N)/VOL;       MID=100*(HIGH+LOW-REF(HIGH+LOW,1))/(HIGH+LOW)
    EMV=MA(MID*VOLUME*(HIGH-LOW)/MA(HIGH-LOW,N),N);    MAEMV=MA(EMV,M)
    return EMV,MAEMV

ret_sub, err_message = quote_ctx.subscribe(symbol, [SubType.K_1M], subscribe_push=False)
if ret_sub == RET_OK:
	ret, df = quote_ctx.get_cur_kline(symbol, 121, SubType.K_1M)
else:
	print(err_message)
	quote_ctx.close()
	sys.exit()

CLOSE=df.close.values;         OPEN=df.open.values
HIGH=df.high.values;           LOW=df.low.values
VOL=df.volume.values;

EMV_arr, MAEMV_arr = EMV(HIGH,LOW,VOL,N=14,M=9)
df_emv = pd.DataFrame({ 'emv': EMV_arr, 'maemv': MAEMV_arr })
print(df_emv.tail(10))

quote_ctx.close()
sys.exit()
