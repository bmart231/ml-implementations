# Going to implement some code snippets for finance machine learning here. Notes will be elsewhere
import numpy as np 
import pandas as pd
import 
# PCA weights from a risk distribution R
def pcaWeights(cov, riskDist = None, riskTarget = 1.):
    # Following the riskAlloc distribution, match riskTarget
    eVal, eVec = np.linalg.eigh(cov)    # Must be Hermitian 
    indices = eVal.argsort()[::-1]      # sorting in desc order
    eVal, eVec = eVal[indices], eVec[:,indices]
    if riskDist is None:
        riskDist = np.zeros(cov.shape[0])
        riskDist[-1] = 1
    loads = riskTarget*(riskDist/eVal)**.5
    wghts = np.dot(eVec, np.reshape(loads, (-1, 1)))
    # ctr = (loads/riskTarget)**2*eVal # verify riskDist
    return wghts     

# FORM A GAPS SERIES, DETRACT IT FROM PRICES

def getRolledSeries(pathIn, key):
    series = pd.read_hdf(pathIn, key = 'bars/ES_10K')
    series['Time']=pd.to_datetime(series['Time'],format = 'Y%m%d%H%M%S%f')
    series = series.set_index('Time')
    gaps = rollGaps(series)
    for fld in ['Close', 'VWAP']:series[fld] -= gaps
    return series 

# ROLL GAPS HELPER FUNCTION
def rollGaps(series, dictio = {'Instrument':'FUT_CUR_GEN_TICKER', 'Open':'FX_OPEN', 'Close':'PX_LAST'}, matchEnd = True):
    # Compute gaps at each roll, between previous close and next open
    rollDates = series[dictio['Instrument ']].drop_duplicates(keep='first').index
    gaps = series[dictio['Close']]*0
    iloc = list(series.index)
    iloc = [iloc.index(i)-1 for i in rollDates] # index of days prior to roll
    gaps.iloc[rollDates[1:]]=series[dictio['Open']].loc[rollDates[1:]] - series[dictio['Close']].iloc[iloc[1:]].values
    gaps = gaps.cumsum()
    if matchEnd:gaps-=gaps.iloc[-1] # roll backward
    return gaps 

# NON-NEGATIVE ROLLED PRICE SERIES 
filepath = ## filepath goes here
raw = pd.read_csv(filePath, index_col=0, parse_dates=True)
gaps = rollGaps(raw, dictio={'Instrument':'Symbol', 'Open':'Open', 'Close':'Close'})
rolled = raw.copy(deep=True)
for fld in ['Open', 'Close']:rolled[fld]-=gaps
rolled['Returns'] = rolled['Close'].diff()/raw['Close'].shift(1)
rolled['rPrices'] = (1+rolled['Returns']).cumprod()