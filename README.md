# Ex-06-Feature-Transformation
## AIM:
To read the given data and perform Feature Transformation process and save the data to a file.
## EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
## ALGORITHM:
### Step 1:
Read the given Data
### Step 2:
Clean the Data Set using Data Cleaning Process
### Step 3:
Apply Feature Transformation techniques to all the features of the data set
### Step 4:
Print the transformed features
## PROGRAM:
~~~
# NAME: DHARSHINI S
# REG NO: 212220220009

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
~~~
## OUTPUT:
![image](https://user-images.githubusercontent.com/113699377/233906227-4e3eeb71-ff12-4a53-ab63-bb1c05ccb537.png)
![image](https://user-images.githubusercontent.com/113699377/233906244-fa60fc06-bec3-449b-bd93-b212c1079d32.png)
![image](https://user-images.githubusercontent.com/113699377/233906340-f2e3a65e-570e-4e06-9945-41c0b638d714.png)
![image](https://user-images.githubusercontent.com/113699377/233906368-dcfdf74e-4114-43b0-a1f6-1ed91b84077e.png)
![image](https://user-images.githubusercontent.com/113699377/233906394-518baf43-26b9-442e-9492-dc54d7dfe1e0.png)
![image](https://user-images.githubusercontent.com/113699377/233906409-0f7ce3fa-b8f3-4e6d-87cb-2204dae3118a.png)
![image](https://user-images.githubusercontent.com/113699377/233906439-9bb60d20-3782-4f95-b392-31660196878b.png)
![image](https://user-images.githubusercontent.com/113699377/233906471-497c52f4-2d6d-4803-b4ed-e15f25172fa7.png)
![image](https://user-images.githubusercontent.com/113699377/233906551-fee94773-742a-4403-9a06-376b6b49e026.png)
## RESULT:
Thus feature transformation is done for the given dataset.
