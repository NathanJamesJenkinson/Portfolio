# **TikTok Project**

Some Exploratory Data Analysis (EDA) and data visualizations have been requested. The management team asked to see a Python notebook showing data structuring and cleaning, as well as any matplotlib/seaborn visualizations plotted to help us understand the data. At the very least, include a graph comparing claim counts to opinion counts, as well as boxplots of the most important variables (like “video duration,” “video like count,” “video comment count,” and “video view count”) to check for outliers. Also, include a breakdown of “author ban status” counts.

Additionally, the management team has recently asked all EDA to include Tableau visualizations. For this data, create a Tableau dashboard showing a simple claims versus opinions count, as well as stacked bar charts of claims versus opinions for variables like video view counts, video like counts, video share counts, and video download counts.

Include an executive summary of your analysis to share with teammates.

**The purpose** of this project is to conduct exploratory data analysis on a provided data set. Perform further EDA on this data with the aim of learning more about the variables. Of particular interest is information related to what distinguishes claim videos from opinion videos.

**The goal** is to explore the dataset and create visualizations.
<br/>

**Part 1:** Imports, links, and loading

**Part 2:** Data Exploration
*   Data cleaning


**Part 3:** Build visualizations

**Part 4:** Evaluate and share results

# **Visualize a story in Tableau and Python**

1. Identify any outliers:


*   What methods are best for identifying outliers?
*   How do you make the decision to keep or exclude outliers from any future models?

Outliers can often be identified initially with descriptive statistics. If the mean and median are far apart then there could be outliers. If the STDV is large compared to the mean there could be outliers. The first place to look is for minimum and maximum values that are extreme. To find contextual outliers and collective outliers it might be easier to create graphs of data points that can show the a value that is clearly not expected within the apparent pattern (if there is one) or to see where there is a significant grouping of datapoints that appear out of place among the other datapoints.
Other methods that could be used include using boxplots to visualize outliers, DBScan Clustering to detect abnormal groupings, isolation random forests to exaggerate the presence of anomolies, RCF forest is a specialized algorithm for giving each datapoint a score that indicates how likely it is to be an outlier, among others.

### **Imports, links, and loading**
For EDA of the data, import the packages that would be most helpful, such as `pandas`, `numpy`, `matplotlib.pyplot`, and `seaborn`.


```python
# Import packages for data manipulation
import pandas as pd
import numpy as np

# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sb
```

Read in the data and store it as a dataframe object.


```python
# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")
```

### **Data exploration and cleaning**


```python
# Display and examine the first few rows of the dataframe
data.head(10)
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
    <tr>
      <th>5</th>
      <td>6</td>
      <td>claim</td>
      <td>8972200955</td>
      <td>35</td>
      <td>someone shared with me that gross domestic pro...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>336647.0</td>
      <td>175546.0</td>
      <td>62303.0</td>
      <td>4293.0</td>
      <td>1857.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>claim</td>
      <td>4958886992</td>
      <td>16</td>
      <td>someone shared with me that elvis presley has ...</td>
      <td>not verified</td>
      <td>active</td>
      <td>750345.0</td>
      <td>486192.0</td>
      <td>193911.0</td>
      <td>8616.0</td>
      <td>5446.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>claim</td>
      <td>2270982263</td>
      <td>41</td>
      <td>someone shared with me that the best selling s...</td>
      <td>not verified</td>
      <td>active</td>
      <td>547532.0</td>
      <td>1072.0</td>
      <td>50.0</td>
      <td>22.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>claim</td>
      <td>5235769692</td>
      <td>50</td>
      <td>someone shared with me that about half of the ...</td>
      <td>not verified</td>
      <td>active</td>
      <td>24819.0</td>
      <td>10160.0</td>
      <td>1050.0</td>
      <td>53.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>claim</td>
      <td>4660861094</td>
      <td>45</td>
      <td>someone shared with me that it would take a 50...</td>
      <td>verified</td>
      <td>active</td>
      <td>931587.0</td>
      <td>171051.0</td>
      <td>67739.0</td>
      <td>4104.0</td>
      <td>2540.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get the size of the data
np.size(data)
```




    232584




```python
# Get the shape of the data
np.shape(data)
```




    (19382, 12)




```python
# Get basic information about the data
data.info()
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



```python
# Generate a table of descriptive statistics
data.describe()
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



### **Select visualization type(s)**

Based on the distribution of the data it is likely that the following visualizations will be most effective:

* Cumulative line graph
* Histogram
* Box plot
* Bar chart for comparisons among multiple categories
* Heat map for a deeper understanding with many bins
* Scatter plot for combinations of data columns

### **Build visualizations**

#### **video_duration_sec**


```python
# Create a boxplot to visualize distribution of `video_duration_sec`
plt.boxplot(data.video_duration_sec.dropna())
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x7fbb649864d0>,
      <matplotlib.lines.Line2D at 0x7fbb64986a10>],
     'caps': [<matplotlib.lines.Line2D at 0x7fbb64986f50>,
      <matplotlib.lines.Line2D at 0x7fbb649914d0>],
     'boxes': [<matplotlib.lines.Line2D at 0x7fbb649f7ed0>],
     'medians': [<matplotlib.lines.Line2D at 0x7fbb64991a50>],
     'fliers': [<matplotlib.lines.Line2D at 0x7fbb649f7450>],
     'means': []}




![png](output_20_1.png)



```python
# Create a histogram
plt.hist(data.video_duration_sec.dropna())
#The distribution of video lengths is surprisingly uniform.
```




    (array([2131., 1761., 2048., 1749., 2030., 1704., 2080., 1723., 2078.,
            2078.]),
     array([ 5. , 10.5, 16. , 21.5, 27. , 32.5, 38. , 43.5, 49. , 54.5, 60. ]),
     <a list of 10 Patch objects>)




![png](output_21_1.png)


#### **video_view_count**


```python
# Create a boxplot to visualize distribution of `video_view_count`
plt.boxplot(data.video_view_count.dropna())
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x7fbb64839950>,
      <matplotlib.lines.Line2D at 0x7fbb64839e90>],
     'caps': [<matplotlib.lines.Line2D at 0x7fbb64840410>,
      <matplotlib.lines.Line2D at 0x7fbb64840950>],
     'boxes': [<matplotlib.lines.Line2D at 0x7fbb64839390>],
     'medians': [<matplotlib.lines.Line2D at 0x7fbb64840ed0>],
     'fliers': [<matplotlib.lines.Line2D at 0x7fbb64849450>],
     'means': []}




![png](output_23_1.png)



```python
# Create a histogram
plt.hist(data.video_view_count.dropna())
#Video view counts are extremely skewed but become surprisingly uniform outside of the category that we would consider viral videos.
```




    (array([10475.,   961.,   944.,   913.,   972.,   920.,   965.,   991.,
              949.,   994.]),
     array([2.000000e+01, 9.999970e+04, 1.999794e+05, 2.999591e+05,
            3.999388e+05, 4.999185e+05, 5.998982e+05, 6.998779e+05,
            7.998576e+05, 8.998373e+05, 9.998170e+05]),
     <a list of 10 Patch objects>)




![png](output_24_1.png)


#### **video_like_count**


```python
# Create a boxplot to visualize distribution of `video_like_count`
plt.boxplot(data.video_like_count.dropna())
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x7fbb647acf10>,
      <matplotlib.lines.Line2D at 0x7fbb64731490>],
     'caps': [<matplotlib.lines.Line2D at 0x7fbb647319d0>,
      <matplotlib.lines.Line2D at 0x7fbb64731f10>],
     'boxes': [<matplotlib.lines.Line2D at 0x7fbb647ac950>],
     'medians': [<matplotlib.lines.Line2D at 0x7fbb647394d0>],
     'fliers': [<matplotlib.lines.Line2D at 0x7fbb64739a10>],
     'means': []}




![png](output_26_1.png)



```python
# Create a histogram
plt.hist(data.video_like_count.dropna())
#Video like counts are extremely skewedc
```




    (array([12694.,  1775.,  1288.,  1035.,   763.,   580.,   426.,   296.,
              163.,    64.]),
     array([     0.,  65783., 131566., 197349., 263132., 328915., 394698.,
            460481., 526264., 592047., 657830.]),
     <a list of 10 Patch objects>)




![png](output_27_1.png)


#### **video_comment_count**


```python
# Create a boxplot to visualize distribution of `video_comment_count`
plt.boxplot(data.video_comment_count.dropna())
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x7fbb646a9510>,
      <matplotlib.lines.Line2D at 0x7fbb646a9a50>],
     'caps': [<matplotlib.lines.Line2D at 0x7fbb646a9f90>,
      <matplotlib.lines.Line2D at 0x7fbb64630510>],
     'boxes': [<matplotlib.lines.Line2D at 0x7fbb646a1f10>],
     'medians': [<matplotlib.lines.Line2D at 0x7fbb64630a90>],
     'fliers': [<matplotlib.lines.Line2D at 0x7fbb64630fd0>],
     'means': []}




![png](output_29_1.png)



```python
# Create a histogram
plt.hist(data.video_comment_count.dropna())
#Video comment counts are extremely skewed and contain a significant number of flagged fliers.
#This data might look significantly different with the outliers removed.
```




    (array([1.6875e+04, 1.2510e+03, 5.2500e+02, 2.0900e+02, 1.1900e+02,
            5.5000e+01, 3.4000e+01, 9.0000e+00, 5.0000e+00, 2.0000e+00]),
     array([   0. ,  959.9, 1919.8, 2879.7, 3839.6, 4799.5, 5759.4, 6719.3,
            7679.2, 8639.1, 9599. ]),
     <a list of 10 Patch objects>)




![png](output_30_1.png)


#### **video_share_count**


```python
# Create a boxplot to visualize distribution of `video_share_count`
plt.boxplot(data.video_share_count.dropna())
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x7fbb64589690>,
      <matplotlib.lines.Line2D at 0x7fbb64589bd0>],
     'caps': [<matplotlib.lines.Line2D at 0x7fbb64591150>,
      <matplotlib.lines.Line2D at 0x7fbb64591690>],
     'boxes': [<matplotlib.lines.Line2D at 0x7fbb645890d0>],
     'medians': [<matplotlib.lines.Line2D at 0x7fbb64591c10>],
     'fliers': [<matplotlib.lines.Line2D at 0x7fbb64598190>],
     'means': []}




![png](output_32_1.png)



```python
# Create a histogram
plt.hist(data.video_share_count.dropna())
#Video share counts are extremely skewed and contain a significant number of flagged fliers.
#This data might look significantly different with the outliers removed.
```




    (array([1.5127e+04, 1.7460e+03, 9.8600e+02, 5.4000e+02, 3.2900e+02,
            1.8800e+02, 9.7000e+01, 5.0000e+01, 1.5000e+01, 6.0000e+00]),
     array([     0.,  25613.,  51226.,  76839., 102452., 128065., 153678.,
            179291., 204904., 230517., 256130.]),
     <a list of 10 Patch objects>)




![png](output_33_1.png)


#### **video_download_count**


```python
# Create a boxplot to visualize distribution of `video_download_count`
plt.boxplot(data.video_download_count.dropna())
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x7fbb64480690>,
      <matplotlib.lines.Line2D at 0x7fbb64480bd0>],
     'caps': [<matplotlib.lines.Line2D at 0x7fbb64487150>,
      <matplotlib.lines.Line2D at 0x7fbb64487690>],
     'boxes': [<matplotlib.lines.Line2D at 0x7fbb644800d0>],
     'medians': [<matplotlib.lines.Line2D at 0x7fbb64487c10>],
     'fliers': [<matplotlib.lines.Line2D at 0x7fbb6448f190>],
     'means': []}




![png](output_35_1.png)



```python
# Create a histogram
plt.hist(data.video_download_count.dropna())
#Video download counts are extremely skewed and contain a significant number of flagged fliers.
#This data might look significantly different with the outliers removed.
```




    (array([1.4921e+04, 1.8020e+03, 9.9500e+02, 5.7800e+02, 3.5500e+02,
            1.9800e+02, 1.2700e+02, 6.3000e+01, 3.3000e+01, 1.2000e+01]),
     array([    0. ,  1499.4,  2998.8,  4498.2,  5997.6,  7497. ,  8996.4,
            10495.8, 11995.2, 13494.6, 14994. ]),
     <a list of 10 Patch objects>)




![png](output_36_1.png)


#### **Claim status by verification status**


```python
# Create a histogram with four bars: one for each combination of claim status and verification status.
data.groupby(['claim_status','verified_status']).size()
CV = {'Claims_Verified':['Claim Not Verified','Claim Verified', 'Opinion Not Verified', 'Opinion Verified'], 'Count':[9399,209,8485,991]}
#claims vs verification

plt.bar(CV['Claims_Verified'],CV['Count'])
plt.xticks(CV['Claims_Verified'],rotation=20)
#There are far more unverified authors than verified authors.
#Verified authors are many times more likely to post opinions than claims.
#Unverified authors could be somewhat more likely to post claims than opinions but it's close so a hypothesis test would be necessary to prove it.
```




    ([<matplotlib.axis.XTick at 0x7fbb643af3d0>,
      <matplotlib.axis.XTick at 0x7fbb643af150>,
      <matplotlib.axis.XTick at 0x7fbb644a3c10>,
      <matplotlib.axis.XTick at 0x7fbb6438cf50>],
     <a list of 4 Text major ticklabel objects>)




![png](output_38_1.png)


#### **Claim status by author ban status**


```python
# Create a histogram for each claim status and each author ban status.
data.groupby(['claim_status','author_ban_status']).size()

CB = {'Claims_Vs_Bans':['Claims: Active','Claims: Under Review', 'Claims: Banned', 'Opinions: Active', 'Opinions: Under Review','Opinions: Banned'], 'Count':[6566,1603,1439,8817,463,196]}
#claims vs verification

plt.barh(CB['Claims_Vs_Bans'],CB['Count'])
#There are many times more authors making claims in the banned category than the active category.
#There are noticeably more active authors providing opinions than making claims.
#A confidence interval would be necessary to prove this relationship is significant, but it appears significant.
#There are also noticeably more authors under review with claims than with opinions.
```




    <BarContainer object of 6 artists>




![png](output_40_1.png)


#### **Median view counts by ban status**


```python
#Create a bar plot with bars for each the median video views of each author ban status.
data.groupby(['author_ban_status']).median()

BSV = {'Views_by_Ban_Status':['Active','Banned','Under Review'], 'Count':[8616,448201,365245.5]}
#Ban Status Views
plt.bar(BSV['Views_by_Ban_Status'],BSV['Count'])
#View counts for non-active authors are drastically larger than for active authors.
#Based on this insight, views might be a good indicator of claim status.
```




    <BarContainer object of 3 artists>




![png](output_42_1.png)



```python
# Calculate the median view count for claim status.
data.groupby(['claim_status']).median()

CSV = {'Views_by_Claim_Status':['Claim','Opinion'], 'Count':[501555,4953]}
#Ban Status Views
plt.bar(CSV['Views_by_Claim_Status'],CSV['Count'])
#The views of claims are astronomically larger than the views of opinions.
```




    <BarContainer object of 2 artists>




![png](output_43_1.png)


#### **Total views by claim status**


```python
# Create a pie graph the depicts the proportions of total views for claim videos and total views for opinion videos.
data.groupby(['claim_status']).size()
labels = 'Claims','Opinions'
sizes = [9908,9476]
plt.pie(sizes,labels=labels)
#While the total views for opinions and claims are similar
#The median of claims dwarfs that of opinions indicating that these two groups might be skewed in opposite directions.
```




    ([<matplotlib.patches.Wedge at 0x7fbb64202d50>,
      <matplotlib.patches.Wedge at 0x7fbb6420f110>],
     [Text(-0.03850029594001087, 1.0993260331732946, 'Claims'),
      Text(0.03850029594001073, -1.0993260331732946, 'Opinions')])




![png](output_45_1.png)


### **Determine outliers**

The ultimate objective of the TikTok project is to build a model that predicts whether a video is a claim or opinion.

Commonly some outliers might be 1.5 * IQR above the 3rd quartile.

The data is heavily skewed to the right so the outlier threshold is better represented by calculating the **median** value instead of the 3rd quartile for each variable and then adding 1.5 * IQR. This results in a threshold that is much lower than it would be if the 3rd quartile was used.


```python
import scipy.stats as sp

    #total number of values in the column:
#columncount = np.size(data['video_view_count'])
    #total of the values in the column:
#total = np.nansum(data['video_view_count'])
#print('Total:',total)
    #median of each column ignoring nans:
#med = np.nanmedian(data['video_view_count'])
#print('Total:',total,'Median:',med)
    #IQR:
    #q3:
#q3 = np.nanpercentile(data['video_view_count'], 75)
    #q1:
#q1 = np.nanpercentile(data['video_view_count'], 25)
#iqr = q3 - q1
#print('Total:',total,'Median:',med,"IQR:",iqr)
    #The value of the first likely outlier:
#outlierlimit = med + 1.5 * iqr
#print('Total:',total,'Median:',med,"IQR:",iqr,"Outlier Limit:",outlierlimit)
    #Percentile of the first likely outlier:
#outlierpercentile = sp.percentileofscore(data['video_view_count'], outlierlimit)
#print('Total:',total,'Median:',med,"IQR:",iqr,"Outlier Limit:",outlierlimit,"Outlier Percentile:",outlierpercentile)
    #Datapoint of the first likely outlier:
#outliervalue = columncount * outlierpercentile / 100
#print('Total:',total,'Median:',med,"IQR:",iqr,"Outlier Limit:",outlierlimit,"Outlier Value:",outliervalue)
    #Number of points that are likely to be outliers:
#totaloutliers = columncount - outliervalue
#print('Total:',total,'Median:',med,"IQR:",iqr,"Outlier Limit:",outlierlimit,"Outlier Value:",outliervalue,"Total Outliers:",totaloutliers)

from pandas.api.types import is_numeric_dtype
for column in data:
    if is_numeric_dtype(data[column]) is True:
        print('Number of outliers,',column,':',np.size(data[column]) - np.size(data[column]) * sp.percentileofscore(data[column], (np.nanmedian(data[column]) + 1.5 * ( np.nanpercentile(data[column], 75) - np.nanpercentile(data[column], 25) ) ) ) / 100 )
```

    Number of outliers, # : 0.0
    Number of outliers, video_id : 0.0
    Number of outliers, video_duration_sec : 0.0
    Number of outliers, video_view_count : 2641.0
    Number of outliers, video_like_count : 3766.0
    Number of outliers, video_share_count : 4030.0
    Number of outliers, video_download_count : 4031.0
    Number of outliers, video_comment_count : 4180.0


#### **Scatterplot**


```python
# Create a scatterplot of `video_like_count` versus `video_comment_count` according to 'claim_status'
sb.scatterplot(data=data, x=data.video_like_count, y=data.video_comment_count, hue=data.claim_status)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fbb641b2610>




![png](output_49_1.png)



```python
# Create a scatterplot of `video_like_count` versus `video_comment_count` for opinions only
opinion_mask = data[data.claim_status == 'opinion']
sb.scatterplot(data=data, x=opinion_mask.video_like_count, y=opinion_mask.video_comment_count)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fbb641e5950>




![png](output_50_1.png)


Create a scatterplot in Tableau Public. https://public.tableau.com/views/TikTokEDA_16890178963150/TikTokEDA?:language=en-US&:display_count=n&:origin=viz_share_link

### **Results and evaluation**

I have learned that every engagement characteristic of these videos are heavily skewed to the right and that each category likely has between 2000-4000 outliers that could be removed in order to make the data more reliable. This is very significant as even 2000/19382 values is over 10% of all of the data values in this dataset.
With respect to claims vs opinions, the number of opinions is very small among this dataset and only occurs among active users.
