import pandas as pd

def solution(df1, df2, df3):
    df = pd.merge(pd.merge(df1, df2, on = 'emp_id', how = 'inner'), df3, on = 'emp_id', how = 'left')
    
    return df
    pass