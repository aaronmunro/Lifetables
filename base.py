import pandas as pd
import numpy as np

life = pd.read_excel('Australian_Life_Tables_2015-17_Females.xlsx')

columnHeadings = ['Age', 'l(x)', 'd(x)', 'p(x)', 'q(x)', 'mu(x)','e(x)', 'L(x)', 'T(x)']

life.columns = columnHeadings

def Px(x):
    return(life.at[x,"p(x)"])

def Qx(x):
    return(1-Px(x))

def lx(x):
    return(life.at[x,"l(x)"])

# tPx, the probability someone aged exactly x years lives t more years. 

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
        sequence2.append(Px(x=year))
    return(np.prod(sequence2))

print(tPx(5,10), nPx(5,10))

# t deferred Qx: Probability a life aged exactly x dies in the year after t years, or survives to x + t, and dies before x + t + 1

def t_Qx(t,x):
    return(tPx(t=t,x=x)*Qx(x=x+t))

# Generalising t deferred Qx to more than 1 year: Probability a life x survives to x + t years but not x + a, 
# ie. number of deaths from t to a l(x+a)-l(x+t) divided by total number alive at age x. Note: a > t

def ta_Qx(t,a,x):
    return((life.at[(x+t),"l(x)"]-life.at[(x+a),"l(x)"]) / (life.at[(x),"l(x)"]))

# Testing. To test if the generalisation is equal to the case of t + 1, a should be set to a = t + 1.

print(t_Qx(t=5,x=15),ta_Qx(t=5,x=15,a=5+1))

# Test larger gap (a-t) than 1 year for death: Probability a life aged 15 dies between 20 and 25. 

print(ta_Qx(t=5,x=15,a=10))

# Interpolation calculation under the Uniform Distribution of Death assumption.

def udd_tPx(t,x):
    frame = [np.floor(x+t), x+t, np.ceil(x+t)]
    frame2 = [np.floor(x), x, np.ceil(x)]
    numerator = ((lx(frame[0])*(1-((x+t)-frame[0])))+(lx(frame[2])*(1-(frame[2]-(x+t)))))
    denominator = ((lx(frame2[0])*(1-((x)-frame2[0])))+(lx(frame2[2])*(1-(frame2[2]-(x)))))
    return(numerator/denominator)

# Tutorial Calculations: 

print(tPx(40,20),nPx(40,20))

print(tPx(20,50)*Qx(70),t_Qx(20,50),ta_Qx(20,21,50))

print(ta_Qx(20,30,40),(tPx(20,40)-tPx(30,40)))

print(((lx(45)-lx(60))/(lx(30)-lx(60))),((ta_Qx(15,30,30))/(1-tPx(30,30))))

print(udd_tPx(2.1,30.5))

# Querying a value based on a value in a different column of the same row 

print(life['q(x)'].idxmax())

print(life.iloc[life['q(x)'].argmax()])

print(life['e(x)'].max())
