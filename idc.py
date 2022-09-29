﻿import json,sys,csv,time
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import pandas_ta as ta

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 1000)

def isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        pass 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass 
    return False

def macd(list):

	print(len(list));	print(list[0], list[-1])

	df=pd.DataFrame({'close':[], 'short':[], 'long':[], 'dif':[], 'dea':[], 'macd':[]})
	df['close'] = pd.DataFrame(list)
	
	df['short'] = df['close'].ewm(span=12, adjust=False).mean()
	df['long'] = df['close'].ewm(span=26, adjust=False).mean()
	df['dif'] = df['short'] - df['long']
	df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
	df['macd'] = (df['dif'] - df['dea']) * 2

	return(df)
	
def ema_x(list):
	df=pd.DataFrame({'close':[]})
	df['close'] = pd.DataFrame(list)
	
	# df = df.reindex(index=df.index[::-1])
	# df = df.reset_index(drop=True)
	
	EMA = pd.Series(df['close'].ewm(span=20, min_periods=20).mean(), name='EM' + str(20))
	df = df.join(EMA)
	return df
	

def ema_sma_all(list):
	df=pd.DataFrame({'close':[]})
	df['close'] = pd.DataFrame(list)
	
	# df = df.reindex(index=df.index[::-1])
	# df = df.reset_index(drop=True)
	
	MA = pd.Series(df['close'].rolling(window=20).mean(), name='MA20')
	df = df.join(MA)

	df['EM20'] = ta.ema(df["close"], length=20)
	df = df.sort_index()


	return df
	

def rsi(list):
	df=pd.DataFrame({'RSI6':[], 'RSI12':[], 'RSI24':[]})
	df = pd.DataFrame({})
	# A = pd.Series(rsix(list,6),name='RSI6')
	# B = pd.Series(rsix(list,12),name='RSI12')
	# C = pd.Series(rsix(list,24),name='RSI24')
	
	
	# df = df.join(A)
	# df = df.join(B)
	# df = df.join(C)
	
	df=pd.DataFrame({'RSI6':rsix(list,6), 'RSI12':rsix(list,12), 'RSI24':rsix(list,24)})
	
	return(df)

def rsix(t, periods):
    length = len(t)
    print(length)
    rsies = [np.nan]*length
    if length <= periods:
        return rsies
    up_avg = 0
    down_avg = 0
    first_t = t[:periods+1]
    for i in range(1, len(first_t)):
        if first_t[i] >= first_t[i-1]:
            up_avg += first_t[i] - first_t[i-1]
        else:
            down_avg += first_t[i-1] - first_t[i]
    up_avg = up_avg / periods
    down_avg = down_avg / periods
    rs = up_avg / down_avg
    rsies[periods] = 100 - 100/(1+rs)

    for j in range(periods+1, length):
        up = 0
        down = 0
        if t[j] >= t[j-1]:
            up = t[j] - t[j-1]
            down = 0
        else:
            up = 0
            down = t[j-1] - t[j]
        up_avg = (up_avg*(periods - 1) + up)/periods
        down_avg = (down_avg*(periods - 1) + down)/periods
        rs = up_avg/down_avg
        rsies[j] = 100 - 100/(1+rs)
    return rsies
	
	
def boll(list):

	df=pd.DataFrame({'close':list})
	
	df['20sma'] = df['close'].rolling(window=20).mean()
	df['stddev'] = df['close'].rolling(window=20).std()

	df['20ema'] =df['close'].ewm(span=20, min_periods=20).mean()
	
	df['bbl'] = df['20sma'] - (2 * df['stddev'])
	df['bbh'] = df['20sma'] + (2 * df['stddev'])
	
	df['bbl2'] = df['20ema'] - (2 * df['stddev'])
	df['bbh2'] = df['20ema'] + (2 * df['stddev'])
	
	return(df)

# def save_all(df1, df2):	# df['EM20']	 df['close']
	
	# if df1['EM20'].iloc[-2] > df1['close'].iloc[-2] and df1['EM20'].iloc[-1] < df1['close'].iloc[-1] :
		# return 1
		
	# if df1['EM20'].iloc[-2] < df1['close'].iloc[-2] and df1['EM20'].iloc[-1] > df1['close'].iloc[-1] :
		# return -1
		
	# return None
	
# result = save_all(df_ema, df_close)
# if result is None:
	# print(' no result' )
	
# elif result == 1:
	# print(' golden cross')
	
# elif result == -1:

	# print(' death cross')
	
	
# 以下函数是我花费3日，重新把所有指标需要用到的细分函数全部制作出来。

# 可做备用 以及自由组合 等。

def RD(N,D=3):   
	return np.round(N,D)
	
def RET(S,N=1):  
	return np.array(S)[-N]      #返回序列倒数第N个值,默认返回最后一个
	
def ABS(S):      
	return np.abs(S)
def MAX(S1,S2):  
	return np.maximum(S1,S2)    #序列max
def MIN(S1,S2):  
	return np.minimum(S1,S2)    #序列min
         
def MA(S,N):           #求序列的N日平均值，返回序列                    
    return pd.Series(S).rolling(N).mean().values

def REF(S, N=1):       #对序列整体下移动N,返回序列(shift后会产生NAN)    
    return pd.Series(S).shift(N).values

def DIFF(S, N=1):      #前一个值减后一个值,前面会产生nan 
    return pd.Series(S).diff(N)  #np.diff(S)直接删除nan，会少一行

def STD(S,N):           #求序列的N日标准差，返回序列    
    return  pd.Series(S).rolling(N).std(ddof=0).values     

def IF(S_BOOL,S_TRUE,S_FALSE):          #序列布尔判断 res=S_TRUE if S_BOOL==True  else  S_FALSE
    return np.where(S_BOOL, S_TRUE, S_FALSE)

def SUM(S, N):            #对序列求N天累计和，返回序列    N=0对序列所有依次求和         
    return pd.Series(S).rolling(N).sum().values if N>0 else pd.Series(S).cumsum()    #pd.rolling_sum(S,N)  (Python2)

def HHV(S,N):             # HHV(C, 5)  # 最近5天收盘最高价        
    return pd.Series(S).rolling(N).max().values      # pd.rolling_max(S,N)  (Python2)

def LLV(S,N):             # LLV(C, 5)  # 最近5天收盘最低价     
    return pd.Series(S).rolling(N).min().values      # pd.rolling_min(S,N)  (Python2)

def EMA(S,N):             #指数移动平均,为了精度 S>4*N  EMA至少需要120周期       
    return pd.Series(S).ewm(span=N, adjust=False).mean().values    

def SMA(S, N, M=1):        #中国式的SMA,至少需要120周期才精确 (雪球180周期)    alpha=1/(1+com)
    return pd.Series(S).ewm(com=N-M, adjust=True).mean().values     

def AVEDEV(S,N):           #平均绝对偏差  (序列与其平均值的绝对差的平均值)   
    avedev=pd.Series(S).rolling(N).apply(lambda x: (np.abs(x - x.mean())).mean())    
    return avedev.values

def SLOPE(S,N,RS=False):    #返S序列N周期回线性回归斜率 (默认只返回斜率,不返回整个直线序列)
    M=pd.Series(S[-N:]);   poly = np.polyfit(M.index, M.values,deg=1);    Y=np.polyval(poly, M.index); 
    if RS: return Y[1]-Y[0],Y
    return Y[1]-Y[0]
	
#------------------   2级：技术指标函数(全部通过0级，1级函数实现） ------------------------------
def MACD(CLOSE,SHORT=12,LONG=26,M=9):            # EMA的关系，S取120日，和雪球小数点2位相同
    DIF = EMA(CLOSE,SHORT)-EMA(CLOSE,LONG);  
    DEA = EMA(DIF,M);      MACD=(DIF-DEA)*2
    return RD(DIF),RD(DEA),RD(MACD)

def KDJ(CLOSE,HIGH,LOW, N=9,M1=3,M2=3):         # KDJ指标
    RSV = (CLOSE - LLV(LOW, N)) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
    K = EMA(RSV, (M1*2-1));    D = EMA(K,(M2*2-1));        J=K*3-D*2
    return K, D, J

def RSI(CLOSE, N=24):      
    DIF = CLOSE-REF(CLOSE,1) 
    return RD(SMA(MAX(DIF,0), N) / SMA(ABS(DIF), N) * 100)  

def WR(CLOSE, HIGH, LOW, N=10, N1=6):            #W&R 威廉指标
    WR = (HHV(HIGH, N) - CLOSE) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
    WR1 = (HHV(HIGH, N1) - CLOSE) / (HHV(HIGH, N1) - LLV(LOW, N1)) * 100
    return RD(WR), RD(WR1)

def BIAS(CLOSE,L1=6, L2=12, L3=24):              # BIAS乖离率
    BIAS1 = (CLOSE - MA(CLOSE, L1)) / MA(CLOSE, L1) * 100
    BIAS2 = (CLOSE - MA(CLOSE, L2)) / MA(CLOSE, L2) * 100
    BIAS3 = (CLOSE - MA(CLOSE, L3)) / MA(CLOSE, L3) * 100
    return RD(BIAS1), RD(BIAS2), RD(BIAS3)

def BOLL(CLOSE,N=20, P=2):                       #BOLL指标，布林带    
    MID = MA(CLOSE, N); 
    UPPER = MID + STD(CLOSE, N) * P
    LOWER = MID - STD(CLOSE, N) * P
    return RD(UPPER), RD(MID), RD(LOWER)    

def PSY(CLOSE,N=12, M=6):  
    PSY=COUNT(CLOSE>REF(CLOSE,1),N)/N*100
    PSYMA=MA(PSY,M)
    return RD(PSY),RD(PSYMA)

def CCI(CLOSE,HIGH,LOW,N=14):  
    TP=(HIGH+LOW+CLOSE)/3
    return (TP-MA(TP,N))/(0.015*AVEDEV(TP,N))
        
def ATR(CLOSE,HIGH,LOW, N=20):                    #真实波动N日平均值
    TR = MAX(MAX((HIGH - LOW), ABS(REF(CLOSE, 1) - HIGH)), ABS(REF(CLOSE, 1) - LOW))
    return MA(TR, N)

def BBI(CLOSE,M1=3,M2=6,M3=12,M4=20):             #BBI多空指标   
    return (MA(CLOSE,M1)+MA(CLOSE,M2)+MA(CLOSE,M3)+MA(CLOSE,M4))/4    

def DMI(CLOSE,HIGH,LOW,M1=14,M2=6):               #动向指标：结果和同花顺，通达信完全一致
    TR = SUM(MAX(MAX(HIGH - LOW, ABS(HIGH - REF(CLOSE, 1))), ABS(LOW - REF(CLOSE, 1))), M1)
    HD = HIGH - REF(HIGH, 1);     LD = REF(LOW, 1) - LOW
    DMP = SUM(IF((HD > 0) & (HD > LD), HD, 0), M1)
    DMM = SUM(IF((LD > 0) & (LD > HD), LD, 0), M1)
    PDI = DMP * 100 / TR;         MDI = DMM * 100 / TR
    ADX = MA(ABS(MDI - PDI) / (PDI + MDI) * 100, M2)
    ADXR = (ADX + REF(ADX, M2)) / 2
    return PDI, MDI, ADX, ADXR  

def TAQ(HIGH,LOW,N):                               #唐安奇通道(海龟)交易指标，大道至简，能穿越牛熊
    UP=HHV(HIGH,N);    DOWN=LLV(LOW,N);    MID=(UP+DOWN)/2
    return UP,MID,DOWN

def KTN(CLOSE,HIGH,LOW,N=20,M=10):                 #肯特纳交易通道, N选20日，ATR选10日
    MID=EMA((HIGH+LOW+CLOSE)/3,N)
    ATRN=ATR(CLOSE,HIGH,LOW,M)
    UPPER=MID+2*ATRN;   LOWER=MID-2*ATRN
    return UPPER,MID,LOWER       
  
def TRIX(CLOSE,M1=12, M2=20):                      #三重指数平滑平均线
    TR = EMA(EMA(EMA(CLOSE, M1), M1), M1)
    TRIX = (TR - REF(TR, 1)) / REF(TR, 1) * 100
    TRMA = MA(TRIX, M2)
    return TRIX, TRMA

def VR(CLOSE,VOL,M1=26):                            #VR容量比率
    LC = REF(CLOSE, 1)
    return SUM(IF(CLOSE > LC, VOL, 0), M1) / SUM(IF(CLOSE <= LC, VOL, 0), M1) * 100

def EMV(HIGH,LOW,VOL,N=14,M=9):                     #简易波动指标 
    VOLUME=MA(VOL,N)/VOL;       MID=100*(HIGH+LOW-REF(HIGH+LOW,1))/(HIGH+LOW)
    EMV=MA(MID*VOLUME*(HIGH-LOW)/MA(HIGH-LOW,N),N);    MAEMV=MA(EMV,M)
    return EMV,MAEMV

def DPO(CLOSE,M1=20, M2=10, M3=6):                  #区间震荡线
    DPO = CLOSE - REF(MA(CLOSE, M1), M2);    MADPO = MA(DPO, M3)
    return DPO, MADPO

def BRAR(OPEN,CLOSE,HIGH,LOW,M1=26):                 #BRAR-ARBR 情绪指标  
    AR = SUM(HIGH - OPEN, M1) / SUM(OPEN - LOW, M1) * 100
    BR = SUM(MAX(0, HIGH - REF(CLOSE, 1)), M1) / SUM(MAX(0, REF(CLOSE, 1) - LOW), M1) * 100
    return AR, BR

def DMA(CLOSE,N1=10,N2=50,M=10):                     #平行线差指标  
    DIF=MA(CLOSE,N1)-MA(CLOSE,N2);    DIFMA=MA(DIF,M)
    return DIF,DIFMA

def MTM(CLOSE,N=12,M=6):                             #动量指标
    MTM=CLOSE-REF(CLOSE,N);         MTMMA=MA(MTM,M)
    return MTM,MTMMA

def MASS(HIGH,LOW,N1=9,N2=25,M=6):                   # 梅斯线
    MASS=SUM(MA(HIGH-LOW,N1)/MA(MA(HIGH-LOW,N1),N1),N2)
    MA_MASS=MA(MASS,M)
    return MASS,MA_MASS
  
def ROC(CLOSE,N=12,M=6):                             #变动率指标
    ROC=100*(CLOSE-REF(CLOSE,N))/REF(CLOSE,N);    MAROC=MA(ROC,M)
    return ROC,MAROC  

def EXPMA(CLOSE,N1=12,N2=50):                        #EMA指数平均数指标
    return EMA(CLOSE,N1),EMA(CLOSE,N2);

def OBV(CLOSE,VOL):                                  #能量潮指标
    return SUM(IF(CLOSE>REF(CLOSE,1),VOL,IF(CLOSE<REF(CLOSE,1),-VOL,0)),0)/10000

def ASI(OPEN,CLOSE,HIGH,LOW,M1=26,M2=10):            #振动升降指标
    LC=REF(CLOSE,1);      AA=ABS(HIGH-LC);     BB=ABS(LOW-LC);
    CC=ABS(HIGH-REF(LOW,1));   DD=ABS(LC-REF(OPEN,1));
    R=IF( (AA>BB) & (AA>CC),AA+BB/2+DD/4,IF( (BB>CC) & (BB>AA),BB+AA/2+DD/4,CC+DD/4));
    X=(CLOSE-LC+(CLOSE-OPEN)/2+LC-REF(OPEN,1));
    SI=16*X/R*MAX(AA,BB);   ASI=SUM(SI,M1);   ASIT=MA(ASI,M2);
    return ASI,ASIT