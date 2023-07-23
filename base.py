import pandas as pd
import numpy as np

life = pd.read_excel('Australian_Life_Tables_2015-17_Males.xlsx')

columnHeadings = ['Age', 'l(x)', 'd(x)', 'p(x)', 'q(x)', 'mu(x)','e(x)', 'L(x)', 'T(x)']

life.columns = columnHeadings

def Px(x):
    return(life.at[x,"p(x)"])

def Qx(x):
    return(1-Px(x))

def tPx(t,x):
    return((life.at[(t+x),"l(x)"]) / (life.at[(x),"l(x)"]))

def tQx(t,x):
    return(1 - tPx(t,x))

# Testing newly made functions 

print(Px(10),Qx(10),tPx(5,10),tQx(5,10))

# Alternative method for tPx: 

def nPx(n,x):
    sequence = list(range(x,x+n))
    sequence2 = list()
    for year in sequence:
        sequence2.append(Px(year))
    return(np.prod(sequence2))

print(tPx(5,10), nPx(5,10))