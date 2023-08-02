# **TikTok Regression Analysis**

Conduct a logistic regression using verified status as the outcome variable.

**Part 1:** EDA & Checking Model Assumptions
* What are some purposes of EDA before constructing a logistic regression model?

**Part 2:** Model Building and Evaluation
* What resources do you find yourself using as you complete this stage?

**Part 3:** Interpreting Model Results

* What key insights emerged from your model(s)?

* What business recommendations do you propose based on the models built?

# **Build a regression model**


```python
# Import packages for data manipulation
import pandas as pd
import numpy as np

# Import packages for data visualization
import matplotlib as plt
import seaborn as sb

# Import packages for data preprocessing


# Import packages for data modeling
import scipy.stats as sp
import sklearn as sk

```


```python
# Load dataset into dataframe
df = pd.read_csv("tiktok_dataset.csv")
```

* What are some purposes of EDA before constructing a logistic regression model?

### **Explore data with EDA**


```python
# Display first few rows
df.head(5)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>claim_status</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>claim</td>
      <td>7017666017</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>claim</td>
      <td>4014381136</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>not verified</td>
      <td>active</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>claim</td>
      <td>9859838091</td>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>not verified</td>
      <td>active</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>claim</td>
      <td>1866847991</td>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>not verified</td>
      <td>active</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>claim</td>
      <td>7105231098</td>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>not verified</td>
      <td>active</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
    </tr>
  </tbody>
</table>
</div>



Get the number of rows and columns in the dataset.


```python
# Get number of rows and columns
np.shape(df)
```




    (19382, 12)



Get the data types of the columns.


```python
# Get data types of columns
df.dtypes
```




    #                             int64
    claim_status                 object
    video_id                      int64
    video_duration_sec            int64
    video_transcription_text     object
    verified_status              object
    author_ban_status            object
    video_view_count            float64
    video_like_count            float64
    video_share_count           float64
    video_download_count        float64
    video_comment_count         float64
    dtype: object



Get basic information about the dataset.


```python
# Get basic information
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19382 entries, 0 to 19381
    Data columns (total 12 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   #                         19382 non-null  int64  
     1   claim_status              19084 non-null  object 
     2   video_id                  19382 non-null  int64  
     3   video_duration_sec        19382 non-null  int64  
     4   video_transcription_text  19084 non-null  object 
     5   verified_status           19382 non-null  object 
     6   author_ban_status         19382 non-null  object 
     7   video_view_count          19084 non-null  float64
     8   video_like_count          19084 non-null  float64
     9   video_share_count         19084 non-null  float64
     10  video_download_count      19084 non-null  float64
     11  video_comment_count       19084 non-null  float64
    dtypes: float64(5), int64(3), object(4)
    memory usage: 1.8+ MB


Generate basic descriptive statistics about the dataset.


```python
# Generate basic descriptive stats
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>19382.000000</td>
      <td>1.938200e+04</td>
      <td>19382.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9691.500000</td>
      <td>5.627454e+09</td>
      <td>32.421732</td>
      <td>254708.558688</td>
      <td>84304.636030</td>
      <td>16735.248323</td>
      <td>1049.429627</td>
      <td>349.312146</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5595.245794</td>
      <td>2.536440e+09</td>
      <td>16.229967</td>
      <td>322893.280814</td>
      <td>133420.546814</td>
      <td>32036.174350</td>
      <td>2004.299894</td>
      <td>799.638865</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.234959e+09</td>
      <td>5.000000</td>
      <td>20.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4846.250000</td>
      <td>3.430417e+09</td>
      <td>18.000000</td>
      <td>4942.500000</td>
      <td>810.750000</td>
      <td>115.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9691.500000</td>
      <td>5.618664e+09</td>
      <td>32.000000</td>
      <td>9954.500000</td>
      <td>3403.500000</td>
      <td>717.000000</td>
      <td>46.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>14536.750000</td>
      <td>7.843960e+09</td>
      <td>47.000000</td>
      <td>504327.000000</td>
      <td>125020.000000</td>
      <td>18222.000000</td>
      <td>1156.250000</td>
      <td>292.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>19382.000000</td>
      <td>9.999873e+09</td>
      <td>60.000000</td>
      <td>999817.000000</td>
      <td>657830.000000</td>
      <td>256130.000000</td>
      <td>14994.000000</td>
      <td>9599.000000</td>
    </tr>
  </tbody>
</table>
</div>



Check for and handle missing values.


```python
# Check for missing values
df.isnull().sum()
```




    #                             0
    claim_status                298
    video_id                      0
    video_duration_sec            0
    video_transcription_text    298
    verified_status               0
    author_ban_status             0
    video_view_count            298
    video_like_count            298
    video_share_count           298
    video_download_count        298
    video_comment_count         298
    dtype: int64




```python
# Drop rows with missing values
dfdrop = df.dropna()
```


```python
# Display first few rows after handling missing values
dfdrop.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>claim_status</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>claim</td>
      <td>7017666017</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>claim</td>
      <td>4014381136</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>not verified</td>
      <td>active</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>claim</td>
      <td>9859838091</td>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>not verified</td>
      <td>active</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>claim</td>
      <td>1866847991</td>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>not verified</td>
      <td>active</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>claim</td>
      <td>7105231098</td>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>not verified</td>
      <td>active</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
    </tr>
  </tbody>
</table>
</div>



Check for and handle duplicates.


```python
# Check for duplicates
df.duplicated().sum()
#df.drop_duplicates()
```




    0



Check for and handle outliers.


```python
# Create a boxplot to visualize distribution of `video_duration_sec`
sb.boxplot(dfdrop.video_duration_sec)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f3831051c50>




![png](output_24_1.png)



```python
# Create a boxplot to visualize distribution of `video_view_count`
sb.boxplot(dfdrop.video_view_count)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f376636cad0>




![png](output_25_1.png)



```python
# Create a boxplot to visualize distribution of `video_like_count`
sb.boxplot(dfdrop.video_like_count)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f37662f8890>




![png](output_26_1.png)



```python
# Create a boxplot to visualize distribution of `video_comment_count`
sb.boxplot(dfdrop.video_comment_count)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f37662c77d0>




![png](output_27_1.png)



```python
df.video_like_count.max()
```




    657830.0




```python
# Check for and handle outliers for video_like_count
for x in ['video_like_count']:
    q75,q25 = np.nanpercentile(df.loc[:,x],[75,25])
    iqr = q75-q25
 
    max = np.nanmedian(df.video_like_count)+(1.5*iqr)
    
    df.loc[df[x] > max,x] = np.nan
```


```python
df.video_like_count.max()
```




    189623.0




```python
# Check class balance for video_comment_count
Vmask = df[df.verified_status == 'verified']
UVmask = df[df.verified_status == 'not verified']

print('Verified video comments:',np.size(Vmask.video_comment_count),'Unverified video commennts:',np.size(UVmask.video_comment_count),sep="\n")
```

    Verified video comments:
    1240
    Unverified video commennts:
    18142



```python
print(df["verified_status"].value_counts())

df.groupby('verified_status').size().plot(kind='pie',
                                       y = "video_comment_count",
                                       label = "Type",
                                       autopct='%1.1f%%')
```

    not verified    18142
    verified         1240
    Name: verified_status, dtype: int64





    <matplotlib.axes._subplots.AxesSubplot at 0x7f3765f3ca90>




![png](output_32_2.png)



```python
# Use resampling to create class balance in the outcome variable, if needed

# Identify data points from majority and minority classes
Vmask = df[df.verified_status == 'verified']
UVmask = df[df.verified_status == 'not verified']

print('Verified video comments:',np.size(Vmask.video_comment_count),'Unverified video commennts:',np.size(UVmask.video_comment_count),sep="\n")

# Upsample the minority class (which is "verified")
from sklearn.utils import resample
Vmaskup = resample(Vmask,
             replace=True,
             n_samples=len(UVmask),
             random_state=42)

print('Verified mask upsampled:',Vmaskup.shape,'Verified video comments upsampled:',np.size(Vmaskup.video_comment_count),'Unverified video commennts:',np.size(UVmask.video_comment_count),sep="\n")

# Combine majority class with upsampled minority class


# Display new class counts

```

    Verified video comments:
    1240
    Unverified video commennts:
    18142
    Verified mask upsampled:
    (18142, 12)
    Verified video comments upsampled:
    18142
    Unverified video commennts:
    18142


Get the average `video_transcription_text` length for videos posted by verified accounts and the average `video_transcription_text` length for videos posted by unverified accounts.




```python
# Extract the length of each `video_transcription_text` and add this as a column to the dataframe
df['VTL'] = df['video_transcription_text'].str.len()
claimVTL = df[df.claim_status == 'claim']
opinionVTL = df[df.claim_status == 'opinion']
# VTL = video text length

# Get the average `video_transcription_text` length for claims and the average `video_transcription_text` length for opinions
meanclaimVTL = np.nanmean(claimVTL.VTL)
meanopinionVTL = np.nanmean(opinionVTL.VTL)
print('Average video text length for claims:',round(meanclaimVTL,1),'Average video text lenth for opinions:',round(meanopinionVTL,1),sep='\n')
```

    Average video text length for claims:
    95.4
    Average video text lenth for opinions:
    82.7



```python
# Display first few rows of dataframe after adding new column
df.head(5)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>claim_status</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>VTL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>claim</td>
      <td>7017666017</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>claim</td>
      <td>4014381136</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>not verified</td>
      <td>active</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
      <td>107.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>claim</td>
      <td>9859838091</td>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>not verified</td>
      <td>active</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
      <td>137.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>claim</td>
      <td>1866847991</td>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>not verified</td>
      <td>active</td>
      <td>437506.0</td>
      <td>NaN</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
      <td>131.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>claim</td>
      <td>7105231098</td>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>not verified</td>
      <td>active</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
      <td>128.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualize the distribution of `video_transcription_text` length for videos posted by verified account
# and videos posted by unverified accounts
VVTL = df[df.verified_status == 'verified']
UVVTL = df[df.verified_status == 'not verified']

from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

axs[0].hist(VVTL.VTL, bins=10)
axs[1].hist(UVVTL.VTL, bins=10)

plt.show()
```


![png](output_37_0.png)


### **Examine correlations**


```python
# Code a correlation matrix to help determine most correlated variables
dfcorr = df.corr()
dfcorr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>VTL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>#</th>
      <td>1.000000</td>
      <td>-0.001714</td>
      <td>-0.000366</td>
      <td>-0.668047</td>
      <td>-0.610972</td>
      <td>-0.451713</td>
      <td>-0.447729</td>
      <td>-0.377445</td>
      <td>-0.226844</td>
    </tr>
    <tr>
      <th>video_id</th>
      <td>-0.001714</td>
      <td>1.000000</td>
      <td>0.009025</td>
      <td>0.000217</td>
      <td>0.004676</td>
      <td>-0.002721</td>
      <td>0.002155</td>
      <td>0.005336</td>
      <td>0.000723</td>
    </tr>
    <tr>
      <th>video_duration_sec</th>
      <td>-0.000366</td>
      <td>0.009025</td>
      <td>1.000000</td>
      <td>0.008481</td>
      <td>-0.002560</td>
      <td>0.011560</td>
      <td>0.013078</td>
      <td>0.000615</td>
      <td>-0.001580</td>
    </tr>
    <tr>
      <th>video_view_count</th>
      <td>-0.668047</td>
      <td>0.000217</td>
      <td>0.008481</td>
      <td>1.000000</td>
      <td>0.719252</td>
      <td>0.665635</td>
      <td>0.664222</td>
      <td>0.554172</td>
      <td>0.230212</td>
    </tr>
    <tr>
      <th>video_like_count</th>
      <td>-0.610972</td>
      <td>0.004676</td>
      <td>-0.002560</td>
      <td>0.719252</td>
      <td>1.000000</td>
      <td>0.827541</td>
      <td>0.835633</td>
      <td>0.698805</td>
      <td>0.213727</td>
    </tr>
    <tr>
      <th>video_share_count</th>
      <td>-0.451713</td>
      <td>-0.002721</td>
      <td>0.011560</td>
      <td>0.665635</td>
      <td>0.827541</td>
      <td>1.000000</td>
      <td>0.679910</td>
      <td>0.574632</td>
      <td>0.147223</td>
    </tr>
    <tr>
      <th>video_download_count</th>
      <td>-0.447729</td>
      <td>0.002155</td>
      <td>0.013078</td>
      <td>0.664222</td>
      <td>0.835633</td>
      <td>0.679910</td>
      <td>1.000000</td>
      <td>0.832464</td>
      <td>0.146382</td>
    </tr>
    <tr>
      <th>video_comment_count</th>
      <td>-0.377445</td>
      <td>0.005336</td>
      <td>0.000615</td>
      <td>0.554172</td>
      <td>0.698805</td>
      <td>0.574632</td>
      <td>0.832464</td>
      <td>1.000000</td>
      <td>0.129659</td>
    </tr>
    <tr>
      <th>VTL</th>
      <td>-0.226844</td>
      <td>0.000723</td>
      <td>-0.001580</td>
      <td>0.230212</td>
      <td>0.213727</td>
      <td>0.147223</td>
      <td>0.146382</td>
      <td>0.129659</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Visualize a correlation heatmap of the data.


```python
# Create a heatmap to visualize correlation among variables
sb.heatmap(dfcorr)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f3765f11d10>




![png](output_41_1.png)


One of the model assumptions for logistic regression is no severe multicollinearity among the features.

View and like, view and share, view and download, like and share, like a download, and download and comment are all possible suspects of multicollinearity. Basically, every engagement variable has at least one relationship with another one.

### **Select variables**

Outcome (Y) variable is claim_status, independent variables are video_duration_sec, video_view_count, video_like_count, video_share_count, video_download_count, video_comment_count, and VTL (video text length).

Select the features.


```python
# Select features using backward elimination via Variance Inflation Factor
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create VTL for dfdrop
dfdrop['VTL'] = dfdrop['video_transcription_text'].str.len()

# Create dummies for claim_status
dfdrop['claim_status'] = dfdrop['claim_status'].map({'opinion':0, 'claim':1})
# 0 for opinion, 1 for claim
  
# Identify independent variables set
X = dfdrop[['video_duration_sec','video_view_count','video_like_count','video_share_count','video_download_count','video_comment_count',
'VTL']]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# Calculate VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)

# Likes and Downloads both have VIFs > 5. Likes will be removed and tested again.
```

                    feature       VIF
    0    video_duration_sec  4.145062
    1      video_view_count  4.690648
    2      video_like_count  9.863807
    3     video_share_count  4.004301
    4  video_download_count  6.812488
    5   video_comment_count  3.883154
    6                   VTL  4.887796



```python
# Reduce independent variables by higest VIF: likes
X = dfdrop[['video_duration_sec','video_view_count','video_share_count','video_download_count','video_comment_count',
'VTL']]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# Calculate VIF for each remaining feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)

# Downloads have a VIF > 5 so it should be removed in another iteration.
```

                    feature       VIF
    0    video_duration_sec  4.145061
    1      video_view_count  3.520326
    2     video_share_count  2.794422
    3  video_download_count  5.604303
    4   video_comment_count  3.882813
    5                   VTL  4.887784



```python
# Reduce independent variables again by higest VIF: likes and downloads
X = dfdrop[['video_duration_sec','video_view_count','video_share_count','video_comment_count',
'VTL']]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# Calculate new VIF for each remaining feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)

# All of the features have a VIF < 5 and are therefore acceptable in a logistic regression.
```

                   feature       VIF
    0   video_duration_sec  4.142992
    1     video_view_count  3.253568
    2    video_share_count  2.567679
    3  video_comment_count  1.930606
    4                  VTL  4.885810



```python
# Get unique values in `claim_status`
dfdrop['claim_status'] = dfdrop['claim_status'].map({0:'opinion', 1:'claim'})
set(dfdrop.claim_status)
# 0 for opinion, 1 for claim
```




    {'claim', 'opinion'}




```python
# Get unique values in `author_ban_status`
set(dfdrop.author_ban_status)
```




    {'active', 'banned', 'under review'}



As shown above, the `claim_status` and `author_ban_status` features are each of data type `object` currently. In order to work with the implementations of models through `sklearn`, these categorical features will need to be made numeric. Encode categorical features in the original dfdrop dataframe using one-hot encoding.

### **Encode variables**


```python
# Display first few rows of the training features that needs to be encoded
print(dfdrop['claim_status'].head(5),dfdrop['author_ban_status'].head(5))
```

    0    claim
    1    claim
    2    claim
    3    claim
    4    claim
    Name: claim_status, dtype: object 0    under review
    1          active
    2          active
    3          active
    4          active
    Name: author_ban_status, dtype: object



```python
# Set up an encoder for one-hot encoding the categorical features
dfdrop['author_ban_status'] = dfdrop['author_ban_status'].map({'under review':'under_review','banned':'banned','active':'active'})
dfdrop['verified_status'] = dfdrop['verified_status'].map({'verified':'verified','not verified':'not_verified'})
dfenc = pd.get_dummies(dfdrop, columns = ['author_ban_status','claim_status','verified_status'])
print(dfenc.head(2))
```

       #    video_id  video_duration_sec  \
    0  1  7017666017                  59   
    1  2  4014381136                  32   
    
                                video_transcription_text  video_view_count  \
    0  someone shared with me that drone deliveries a...          343296.0   
    1  someone shared with me that there are more mic...          140877.0   
    
       video_like_count  video_share_count  video_download_count  \
    0           19425.0              241.0                   1.0   
    1           77355.0            19034.0                1161.0   
    
       video_comment_count  VTL  author_ban_status_active  \
    0                  0.0   97                         0   
    1                684.0  107                         1   
    
       author_ban_status_banned  author_ban_status_under_review  \
    0                         0                               1   
    1                         0                               0   
    
       claim_status_claim  claim_status_opinion  verified_status_not_verified  \
    0                   1                     0                             1   
    1                   1                     0                             1   
    
       verified_status_verified  
    0                         0  
    1                         0  



```python
# Fit and transform the training features using the encoder
X = dfenc[['video_duration_sec','video_view_count','video_share_count','video_comment_count',
'VTL','author_ban_status_active','author_ban_status_banned','author_ban_status_under_review','claim_status_claim']]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# Calculate new VIF for each remaining feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)

# Note that including claim_status_opinion results in an error because claims and opinions
# are perfect complements of each other within the claim_status column so only claim_status_claim is tested here.
```

                              feature        VIF
    0              video_duration_sec   1.000651
    1                video_view_count   3.377663
    2               video_share_count   2.017365
    3             video_comment_count   1.621337
    4                             VTL   1.103550
    5        author_ban_status_active  19.148064
    6        author_ban_status_banned   3.164011
    7  author_ban_status_under_review   3.589625
    8              claim_status_claim   2.653298



```python
# Fit and transform the training features using the encoder: remove author_ban_status_active
X = dfenc[['video_duration_sec','video_view_count','video_share_count','video_comment_count',
'VTL','author_ban_status_banned','author_ban_status_under_review','claim_status_claim']]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# Calculate new VIF for each remaining feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)
```

                              feature       VIF
    0              video_duration_sec  4.146030
    1                video_view_count  5.479390
    2               video_share_count  2.567876
    3             video_comment_count  1.930746
    4                             VTL  5.492152
    5        author_ban_status_banned  1.184359
    6  author_ban_status_under_review  1.190507
    7              claim_status_claim  5.324973



```python
# Fit and transform the training features using the encoder: remove author_ban_status_active and claim_status_claim
X = dfenc[['video_duration_sec','video_view_count','video_share_count','video_comment_count',
'VTL','author_ban_status_banned','author_ban_status_under_review']]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# Calculate new VIF for each remaining feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)

# This results in VIFs < 5, so these are the best variables to use as they do not have high multicollinearity.
```

                              feature       VIF
    0              video_duration_sec  4.143068
    1                video_view_count  3.345364
    2               video_share_count  2.567874
    3             video_comment_count  1.930620
    4                             VTL  4.991513
    5        author_ban_status_banned  1.154464
    6  author_ban_status_under_review  1.167685



```python
# Get feature names from encoder and display first few rows of encoded training features
print(dfenc[['video_duration_sec','video_view_count','video_share_count','video_comment_count',
'VTL','author_ban_status_banned','author_ban_status_under_review']].head(2))
```

       video_duration_sec  video_view_count  video_share_count  \
    0                  59          343296.0              241.0   
    1                  32          140877.0            19034.0   
    
       video_comment_count  VTL  author_ban_status_banned  \
    0                  0.0   97                         0   
    1                684.0  107                         0   
    
       author_ban_status_under_review  
    0                               1  
    1                               0  



```python
# Place encoded training features (which is currently an array) into a dataframe
dfdrop['author_ban_status_banned'] = dfenc['author_ban_status_banned']
dfdrop['author_ban_status_under_review'] = dfenc['author_ban_status_under_review']
dfdrop['author_ban_status_active'] = dfenc['author_ban_status_active']
dfdrop['claim_status_claim'] = dfenc['claim_status_claim']
dfdrop['verified_status_verified'] = dfenc['verified_status_verified']


# Display first few rows
dfdrop.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>claim_status</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>VTL</th>
      <th>author_ban_status_banned</th>
      <th>author_ban_status_under_review</th>
      <th>author_ban_status_active</th>
      <th>claim_status_claim</th>
      <th>verified_status_verified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>claim</td>
      <td>7017666017</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>not_verified</td>
      <td>under_review</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>97</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>claim</td>
      <td>4014381136</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>not_verified</td>
      <td>active</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
      <td>107</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Display first few rows of `X_train` with `claim_status` and `author_ban_status` columns dropped
#(since these features are being transformed to numeric)
#print(dfdrop[['video_duration_sec','video_view_count','video_share_count','video_comment_count','VTL','author_ban_status_banned','author_ban_status_under_review']].head(2))

dfdrop.loc[:, ~dfdrop.columns.isin(['claim_status', 'author_ban_status'])]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>VTL</th>
      <th>author_ban_status_banned</th>
      <th>author_ban_status_under_review</th>
      <th>author_ban_status_active</th>
      <th>claim_status_claim</th>
      <th>verified_status_verified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>7017666017</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>not_verified</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>97</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4014381136</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>not_verified</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
      <td>107</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>9859838091</td>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>not_verified</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
      <td>137</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1866847991</td>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>not_verified</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
      <td>131</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>7105231098</td>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>not_verified</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
      <td>128</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19079</th>
      <td>19080</td>
      <td>1492320297</td>
      <td>49</td>
      <td>in our opinion the earth holds about 11 quinti...</td>
      <td>not_verified</td>
      <td>6067.0</td>
      <td>423.0</td>
      <td>81.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>65</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19080</th>
      <td>19081</td>
      <td>9841347807</td>
      <td>23</td>
      <td>in our opinion the queens in ant colonies live...</td>
      <td>not_verified</td>
      <td>2973.0</td>
      <td>820.0</td>
      <td>70.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>66</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19081</th>
      <td>19082</td>
      <td>8024379946</td>
      <td>50</td>
      <td>in our opinion the moon is moving away from th...</td>
      <td>not_verified</td>
      <td>734.0</td>
      <td>102.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>53</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19082</th>
      <td>19083</td>
      <td>7425795014</td>
      <td>8</td>
      <td>in our opinion lightning strikes somewhere on ...</td>
      <td>not_verified</td>
      <td>3394.0</td>
      <td>655.0</td>
      <td>123.0</td>
      <td>11.0</td>
      <td>4.0</td>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19083</th>
      <td>19084</td>
      <td>4094655375</td>
      <td>58</td>
      <td>in our opinion a pineapple plant can only prod...</td>
      <td>not_verified</td>
      <td>5034.0</td>
      <td>815.0</td>
      <td>281.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>70</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>19084 rows × 16 columns</p>
</div>



### **Train-test split**


```python
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X = dfdrop[['video_duration_sec','video_view_count','video_like_count','video_share_count','video_download_count','video_comment_count',
'VTL','author_ban_status_banned','author_ban_status_under_review','author_ban_status_active','claim_status_claim']]
y = dfdrop['verified_status_verified']
  
# using the train test split function
X_train, X_test, y_train, y_test = train_test_split(X,y ,
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)
  
# print out train and test sets
print('X_train: ')
print(X_train.head(3))
print('')
print('X_test: ')
print(X_test.head(3))
print('')
print('y_train:')
print(y_train.head(3))
print('')
print('y_test: ')
print(y_test.head(3))
```

    X_train: 
          video_duration_sec  video_view_count  video_like_count  \
    1280                   6          505712.0          226924.0   
    1089                  44          613997.0          121833.0   
    2569                  47          641644.0          391850.0   
    
          video_share_count  video_download_count  video_comment_count  VTL  \
    1280            20502.0                2893.0                593.0  100   
    1089            24182.0                1624.0                269.0   70   
    2569           111575.0                6305.0               2035.0   96   
    
          author_ban_status_banned  author_ban_status_under_review  \
    1280                         0                               1   
    1089                         1                               0   
    2569                         0                               0   
    
          author_ban_status_active  claim_status_claim  
    1280                         0                   1  
    1089                         0                   1  
    2569                         1                   1  
    
    X_test: 
           video_duration_sec  video_view_count  video_like_count  \
    8404                   32          655785.0          202711.0   
    12991                  28            5757.0             455.0   
    3417                   23          846593.0          545364.0   
    
           video_share_count  video_download_count  video_comment_count  VTL  \
    8404              7433.0                4106.0               2271.0   76   
    12991               58.0                   8.0                  3.0  101   
    3417             42075.0               12866.0               7694.0   90   
    
           author_ban_status_banned  author_ban_status_under_review  \
    8404                          0                               0   
    12991                         0                               0   
    3417                          1                               0   
    
           author_ban_status_active  claim_status_claim  
    8404                          1                   1  
    12991                         1                   0  
    3417                          0                   1  
    
    y_train:
    1280    0
    1089    0
    2569    0
    Name: verified_status_verified, dtype: uint8
    
    y_test: 
    8404     0
    12991    0
    3417     0
    Name: verified_status_verified, dtype: uint8



```python
y_train
```




    1280     0
    1089     0
    2569     0
    1486     0
    15535    0
            ..
    14180    0
    7896     0
    6310     0
    17113    0
    8261     0
    Name: verified_status_verified, Length: 14313, dtype: uint8




```python
data_final_vars=dfenc.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(X_train, y_train.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

# It is interesting that recursive feature elimination does not remove
# the columns of likes, downloads, author_ban_status_active, and claim_status_claim.
# They already have been proven to have high multicollinearity so they cannot be used in this model though.
```

    [ True  True  True  True  True  True  True  True  True  True  True]
    [1 1 1 1 1 1 1 1 1 1 1]


Confirm that the dimensions of the training and testing sets are in alignment.


```python
# Get shape of each training and testing set
print('X_train: ')
print(np.shape(X_train))
print('')
print('X_test: ')
print(np.shape(X_test))
print('')
print('y_train:')
print(np.shape(y_train))
print('')
print('y_test:')
print(np.shape(y_test))
```

    X_train: 
    (14313, 11)
    
    X_test: 
    (4771, 11)
    
    y_train:
    (14313,)
    
    y_test:
    (4771,)



```python
# Check data types
print('X_train : ')
print(X_train.dtypes)
print('')
print('X_test : ')
print(X_test.dtypes)
print('')
print('y_train:')
print(y_train.dtypes)
print('')
print('y_test:')
print(y_test.dtypes)
```

    X_train : 
    video_duration_sec                  int64
    video_view_count                  float64
    video_like_count                  float64
    video_share_count                 float64
    video_download_count              float64
    video_comment_count               float64
    VTL                                 int64
    author_ban_status_banned            uint8
    author_ban_status_under_review      uint8
    author_ban_status_active            uint8
    claim_status_claim                  uint8
    dtype: object
    
    X_test : 
    video_duration_sec                  int64
    video_view_count                  float64
    video_like_count                  float64
    video_share_count                 float64
    video_download_count              float64
    video_comment_count               float64
    VTL                                 int64
    author_ban_status_banned            uint8
    author_ban_status_under_review      uint8
    author_ban_status_active            uint8
    claim_status_claim                  uint8
    dtype: object
    
    y_train:
    uint8
    
    y_test:
    uint8



```python
# Get unique values of outcome variable
print('X_train:')
print(set(X_train))
print('')
print('X_test:')
print(set(X_test))
print('')
print('y_train:')
print(set(y_train))
print('')
print('y_test:')
print(set(y_test))
```

    X_train:
    {'video_download_count', 'author_ban_status_active', 'claim_status_claim', 'video_share_count', 'author_ban_status_banned', 'VTL', 'video_comment_count', 'video_view_count', 'video_like_count', 'video_duration_sec', 'author_ban_status_under_review'}
    
    X_test:
    {'video_download_count', 'author_ban_status_active', 'claim_status_claim', 'video_share_count', 'author_ban_status_banned', 'VTL', 'video_comment_count', 'video_view_count', 'video_like_count', 'video_duration_sec', 'author_ban_status_under_review'}
    
    y_train:
    {0, 1}
    
    y_test:
    {0, 1}


### **Model building**


```python
# Construct a logistic regression model and fit it to the training set
cols = ['video_duration_sec','video_view_count','video_share_count','video_comment_count',
'VTL','author_ban_status_banned','author_ban_status_under_review']
X=X_train[cols]
y=y_train

import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

# the P-values of share and comment are > .05 so they need to be removed.
```

    Optimization terminated successfully.
             Current function value: inf
             Iterations 8


    /opt/conda/lib/python3.7/site-packages/statsmodels/base/model.py:548: HessianInversionWarning: Inverting hessian failed, no bse or cov_params available
      'available', HessianInversionWarning)
    /opt/conda/lib/python3.7/site-packages/statsmodels/base/model.py:548: HessianInversionWarning: Inverting hessian failed, no bse or cov_params available
      'available', HessianInversionWarning)


                                    Results: Logit
    ===============================================================================
    Model:                  Logit                       Pseudo R-squared:    inf   
    Dependent Variable:     verified_status_verified    AIC:                 inf   
    Date:                   2023-07-28 18:37            BIC:                 inf   
    No. Observations:       14313                       Log-Likelihood:      -inf  
    Df Model:               6                           LL-Null:             0.0000
    Df Residuals:           14306                       LLR p-value:         1.0000
    Converged:              1.0000                      Scale:               1.0000
    No. Iterations:         8.0000                                                 
    -------------------------------------------------------------------------------
                                    Coef.  Std.Err.    z     P>|z|   [0.025  0.975]
    -------------------------------------------------------------------------------
    video_duration_sec             -0.0108   0.0019  -5.5329 0.0000 -0.0146 -0.0069
    video_view_count               -0.0000   0.0000  -8.7712 0.0000 -0.0000 -0.0000
    video_share_count               0.0000   0.0000   1.0666 0.2862 -0.0000  0.0000
    video_comment_count             0.0000   0.0001   0.4952 0.6205 -0.0001  0.0002
    VTL                            -0.0225   0.0009 -26.3884 0.0000 -0.0242 -0.0208
    author_ban_status_banned       -0.4997   0.1795  -2.7842 0.0054 -0.8515 -0.1479
    author_ban_status_under_review -0.5365   0.1525  -3.5180 0.0004 -0.8354 -0.2376
    ===============================================================================
    



```python
# Construct a logistic regression model and fit it to the training set
cols = ['video_duration_sec','video_view_count',
'VTL','author_ban_status_banned','author_ban_status_under_review']
X=X_train[cols]
y=y_train

import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

# all of the P-values are < .05, so this is the final model.
```

    Optimization terminated successfully.
             Current function value: inf
             Iterations 8


    /opt/conda/lib/python3.7/site-packages/statsmodels/base/model.py:548: HessianInversionWarning: Inverting hessian failed, no bse or cov_params available
      'available', HessianInversionWarning)


                                    Results: Logit
    ===============================================================================
    Model:                  Logit                       Pseudo R-squared:    inf   
    Dependent Variable:     verified_status_verified    AIC:                 inf   
    Date:                   2023-07-28 18:38            BIC:                 inf   
    No. Observations:       14313                       Log-Likelihood:      -inf  
    Df Model:               4                           LL-Null:             0.0000
    Df Residuals:           14308                       LLR p-value:         1.0000
    Converged:              1.0000                      Scale:               1.0000
    No. Iterations:         8.0000                                                 
    -------------------------------------------------------------------------------
                                    Coef.  Std.Err.    z     P>|z|   [0.025  0.975]
    -------------------------------------------------------------------------------
    video_duration_sec             -0.0107   0.0019  -5.5261 0.0000 -0.0146 -0.0069
    video_view_count               -0.0000   0.0000 -11.8161 0.0000 -0.0000 -0.0000
    VTL                            -0.0225   0.0009 -26.4119 0.0000 -0.0242 -0.0208
    author_ban_status_banned       -0.4993   0.1795  -2.7822 0.0054 -0.8511 -0.1476
    author_ban_status_under_review -0.5370   0.1525  -3.5213 0.0004 -0.8359 -0.2381
    ===============================================================================
    


    /opt/conda/lib/python3.7/site-packages/statsmodels/base/model.py:548: HessianInversionWarning: Inverting hessian failed, no bse or cov_params available
      'available', HessianInversionWarning)



```python
# Build regression model
clf = LogisticRegression().fit(X_train,y_train)

# Save predictions
y_pred = clf.predict(X_test)
```

### **Results and evaluation**

Test the logistic regression model. Use the model to make predictions on the encoded testing set.


```python
# Use the logistic regression model to get predictions on the encoded testing set
# Display the predictions on the encoded testing set
y_pred = clf.predict(X_test)
y_pred
```




    array([0, 0, 0, ..., 0, 0, 0], dtype=uint8)




```python
# Display the labels of the testing set
y_test
# 0 is not verified, 1 is verified
```




    8404     0
    12991    0
    3417     0
    13501    0
    6799     0
            ..
    5373     0
    7733     0
    6285     0
    8963     0
    6381     0
    Name: verified_status_verified, Length: 4771, dtype: uint8



Confirm again that the dimensions of the training and testing sets are in alignment since additional features were added.


```python
# Get shape of each training and testing set
print('X_train: ')
print(np.shape(X_train))
print('')
print('X_test: ')
print(np.shape(X_test))
print('')
print('y_train:')
print(np.shape(y_train))
print('')
print('y_test:')
print(np.shape(y_test))
```

    X_train: 
    (14313, 11)
    
    X_test: 
    (4771, 11)
    
    y_train:
    (14313,)
    
    y_test:
    (4771,)


### **Visualize model results**

Create a confusion matrix to visualize the results of the logistic regression model.


```python
# Compute values for confusion matrix
import sklearn.metrics as metrics
cm = metrics.confusion_matrix(y_test, y_pred, labels = clf.classes_)

# Create display of confusion matrix
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = clf.classes_)

# Plot confusion matrix


# Display plot
disp.plot()
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7f3763236610>




![png](output_81_1.png)


Create a classification report that includes precision, recall, f1-score, and accuracy metrics to evaluate the performance of the logistic regression model.


```python
# Create a classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.94      1.00      0.97      4463
               1       0.00      0.00      0.00       308
    
        accuracy                           0.94      4771
       macro avg       0.47      0.50      0.48      4771
    weighted avg       0.88      0.94      0.90      4771
    


### **Interpret model coefficients**


```python
# Get the feature names from the model and the model coefficients (which represent log-odds ratios)
# Place into a DataFrame for readability
print('Feature names: ',X_train.columns,'Model coefficients: ',clf.coef_,sep='\n')
```

    Feature names: 
    Index(['video_duration_sec', 'video_view_count', 'video_like_count',
           'video_share_count', 'video_download_count', 'video_comment_count',
           'VTL', 'author_ban_status_banned', 'author_ban_status_under_review',
           'author_ban_status_active', 'claim_status_claim'],
          dtype='object')
    Model coefficients: 
    [[-9.07344751e-03 -2.15101616e-06 -7.47223629e-07  6.40727602e-06
      -2.28591264e-04  4.23643750e-04 -2.36470385e-02 -1.15612199e-05
      -1.98523304e-05 -2.52553836e-04 -3.91581928e-05]]


### **Conclusion**

1. What are the key takeaways from this project?
This model is not useful and needs to be attempted again.

2. What results can be presented from this project?
More information is needed to predict verification status, or errors were made during the model building process.


```python

```
