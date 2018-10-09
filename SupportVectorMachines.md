

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
```


```python
#We use the built in data set for breast cancer from sklearn
from sklearn.datasets import load_breast_cancer
```


```python
cancer= load_breast_cancer()
```


```python
cancer.keys()
```




    ['target_names', 'data', 'target', 'DESCR', 'feature_names']




```python
#In order to get a detailed description we perform the following 
print(cancer)
```

    {'target_names': array(['malignant', 'benign'], dtype='|S9'), 'data': array([[1.799e+01, 1.038e+01, 1.228e+02, ..., 2.654e-01, 4.601e-01,
            1.189e-01],
           [2.057e+01, 1.777e+01, 1.329e+02, ..., 1.860e-01, 2.750e-01,
            8.902e-02],
           [1.969e+01, 2.125e+01, 1.300e+02, ..., 2.430e-01, 3.613e-01,
            8.758e-02],
           ...,
           [1.660e+01, 2.808e+01, 1.083e+02, ..., 1.418e-01, 2.218e-01,
            7.820e-02],
           [2.060e+01, 2.933e+01, 1.401e+02, ..., 2.650e-01, 4.087e-01,
            1.240e-01],
           [7.760e+00, 2.454e+01, 4.792e+01, ..., 0.000e+00, 2.871e-01,
            7.039e-02]]), 'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,
           1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,
           1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
           1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0,
           0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,
           1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0,
           0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
           1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1,
           1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,
           0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,
           0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
           1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
           1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1,
           1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
           1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
           1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
           1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]), 'DESCR': 'Breast Cancer Wisconsin (Diagnostic) Database\n=============================================\n\nNotes\n-----\nData Set Characteristics:\n    :Number of Instances: 569\n\n    :Number of Attributes: 30 numeric, predictive attributes and the class\n\n    :Attribute Information:\n        - radius (mean of distances from center to points on the perimeter)\n        - texture (standard deviation of gray-scale values)\n        - perimeter\n        - area\n        - smoothness (local variation in radius lengths)\n        - compactness (perimeter^2 / area - 1.0)\n        - concavity (severity of concave portions of the contour)\n        - concave points (number of concave portions of the contour)\n        - symmetry \n        - fractal dimension ("coastline approximation" - 1)\n\n        The mean, standard error, and "worst" or largest (mean of the three\n        largest values) of these features were computed for each image,\n        resulting in 30 features.  For instance, field 3 is Mean Radius, field\n        13 is Radius SE, field 23 is Worst Radius.\n\n        - class:\n                - WDBC-Malignant\n                - WDBC-Benign\n\n    :Summary Statistics:\n\n    ===================================== ====== ======\n                                           Min    Max\n    ===================================== ====== ======\n    radius (mean):                        6.981  28.11\n    texture (mean):                       9.71   39.28\n    perimeter (mean):                     43.79  188.5\n    area (mean):                          143.5  2501.0\n    smoothness (mean):                    0.053  0.163\n    compactness (mean):                   0.019  0.345\n    concavity (mean):                     0.0    0.427\n    concave points (mean):                0.0    0.201\n    symmetry (mean):                      0.106  0.304\n    fractal dimension (mean):             0.05   0.097\n    radius (standard error):              0.112  2.873\n    texture (standard error):             0.36   4.885\n    perimeter (standard error):           0.757  21.98\n    area (standard error):                6.802  542.2\n    smoothness (standard error):          0.002  0.031\n    compactness (standard error):         0.002  0.135\n    concavity (standard error):           0.0    0.396\n    concave points (standard error):      0.0    0.053\n    symmetry (standard error):            0.008  0.079\n    fractal dimension (standard error):   0.001  0.03\n    radius (worst):                       7.93   36.04\n    texture (worst):                      12.02  49.54\n    perimeter (worst):                    50.41  251.2\n    area (worst):                         185.2  4254.0\n    smoothness (worst):                   0.071  0.223\n    compactness (worst):                  0.027  1.058\n    concavity (worst):                    0.0    1.252\n    concave points (worst):               0.0    0.291\n    symmetry (worst):                     0.156  0.664\n    fractal dimension (worst):            0.055  0.208\n    ===================================== ====== ======\n\n    :Missing Attribute Values: None\n\n    :Class Distribution: 212 - Malignant, 357 - Benign\n\n    :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian\n\n    :Donor: Nick Street\n\n    :Date: November, 1995\n\nThis is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.\nhttps://goo.gl/U2Uwz2\n\nFeatures are computed from a digitized image of a fine needle\naspirate (FNA) of a breast mass.  They describe\ncharacteristics of the cell nuclei present in the image.\n\nSeparating plane described above was obtained using\nMultisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree\nConstruction Via Linear Programming." Proceedings of the 4th\nMidwest Artificial Intelligence and Cognitive Science Society,\npp. 97-101, 1992], a classification method which uses linear\nprogramming to construct a decision tree.  Relevant features\nwere selected using an exhaustive search in the space of 1-4\nfeatures and 1-3 separating planes.\n\nThe actual linear program used to obtain the separating plane\nin the 3-dimensional space is that described in:\n[K. P. Bennett and O. L. Mangasarian: "Robust Linear\nProgramming Discrimination of Two Linearly Inseparable Sets",\nOptimization Methods and Software 1, 1992, 23-34].\n\nThis database is also available through the UW CS ftp server:\n\nftp ftp.cs.wisc.edu\ncd math-prog/cpo-dataset/machine-learn/WDBC/\n\nReferences\n----------\n   - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction \n     for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on \n     Electronic Imaging: Science and Technology, volume 1905, pages 861-870,\n     San Jose, CA, 1993.\n   - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and \n     prognosis via linear programming. Operations Research, 43(4), pages 570-577, \n     July-August 1995.\n   - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques\n     to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) \n     163-171.\n', 'feature_names': array(['mean radius', 'mean texture', 'mean perimeter', 'mean area',
           'mean smoothness', 'mean compactness', 'mean concavity',
           'mean concave points', 'mean symmetry', 'mean fractal dimension',
           'radius error', 'texture error', 'perimeter error', 'area error',
           'smoothness error', 'compactness error', 'concavity error',
           'concave points error', 'symmetry error',
           'fractal dimension error', 'worst radius', 'worst texture',
           'worst perimeter', 'worst area', 'worst smoothness',
           'worst compactness', 'worst concavity', 'worst concave points',
           'worst symmetry', 'worst fractal dimension'], dtype='|S23')}



```python
#In order to get a detailed description we perform the following 
print(cancer['DESCR'])
```

    Breast Cancer Wisconsin (Diagnostic) Database
    =============================================
    
    Notes
    -----
    Data Set Characteristics:
        :Number of Instances: 569
    
        :Number of Attributes: 30 numeric, predictive attributes and the class
    
        :Attribute Information:
            - radius (mean of distances from center to points on the perimeter)
            - texture (standard deviation of gray-scale values)
            - perimeter
            - area
            - smoothness (local variation in radius lengths)
            - compactness (perimeter^2 / area - 1.0)
            - concavity (severity of concave portions of the contour)
            - concave points (number of concave portions of the contour)
            - symmetry 
            - fractal dimension ("coastline approximation" - 1)
    
            The mean, standard error, and "worst" or largest (mean of the three
            largest values) of these features were computed for each image,
            resulting in 30 features.  For instance, field 3 is Mean Radius, field
            13 is Radius SE, field 23 is Worst Radius.
    
            - class:
                    - WDBC-Malignant
                    - WDBC-Benign
    
        :Summary Statistics:
    
        ===================================== ====== ======
                                               Min    Max
        ===================================== ====== ======
        radius (mean):                        6.981  28.11
        texture (mean):                       9.71   39.28
        perimeter (mean):                     43.79  188.5
        area (mean):                          143.5  2501.0
        smoothness (mean):                    0.053  0.163
        compactness (mean):                   0.019  0.345
        concavity (mean):                     0.0    0.427
        concave points (mean):                0.0    0.201
        symmetry (mean):                      0.106  0.304
        fractal dimension (mean):             0.05   0.097
        radius (standard error):              0.112  2.873
        texture (standard error):             0.36   4.885
        perimeter (standard error):           0.757  21.98
        area (standard error):                6.802  542.2
        smoothness (standard error):          0.002  0.031
        compactness (standard error):         0.002  0.135
        concavity (standard error):           0.0    0.396
        concave points (standard error):      0.0    0.053
        symmetry (standard error):            0.008  0.079
        fractal dimension (standard error):   0.001  0.03
        radius (worst):                       7.93   36.04
        texture (worst):                      12.02  49.54
        perimeter (worst):                    50.41  251.2
        area (worst):                         185.2  4254.0
        smoothness (worst):                   0.071  0.223
        compactness (worst):                  0.027  1.058
        concavity (worst):                    0.0    1.252
        concave points (worst):               0.0    0.291
        symmetry (worst):                     0.156  0.664
        fractal dimension (worst):            0.055  0.208
        ===================================== ====== ======
    
        :Missing Attribute Values: None
    
        :Class Distribution: 212 - Malignant, 357 - Benign
    
        :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian
    
        :Donor: Nick Street
    
        :Date: November, 1995
    
    This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.
    https://goo.gl/U2Uwz2
    
    Features are computed from a digitized image of a fine needle
    aspirate (FNA) of a breast mass.  They describe
    characteristics of the cell nuclei present in the image.
    
    Separating plane described above was obtained using
    Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree
    Construction Via Linear Programming." Proceedings of the 4th
    Midwest Artificial Intelligence and Cognitive Science Society,
    pp. 97-101, 1992], a classification method which uses linear
    programming to construct a decision tree.  Relevant features
    were selected using an exhaustive search in the space of 1-4
    features and 1-3 separating planes.
    
    The actual linear program used to obtain the separating plane
    in the 3-dimensional space is that described in:
    [K. P. Bennett and O. L. Mangasarian: "Robust Linear
    Programming Discrimination of Two Linearly Inseparable Sets",
    Optimization Methods and Software 1, 1992, 23-34].
    
    This database is also available through the UW CS ftp server:
    
    ftp ftp.cs.wisc.edu
    cd math-prog/cpo-dataset/machine-learn/WDBC/
    
    References
    ----------
       - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction 
         for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on 
         Electronic Imaging: Science and Technology, volume 1905, pages 861-870,
         San Jose, CA, 1993.
       - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and 
         prognosis via linear programming. Operations Research, 43(4), pages 570-577, 
         July-August 1995.
       - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
         to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 
         163-171.
    



```python
#The precdition wewant is to see if fthe cancer is malignent or bening 
```


```python
#We create a dataframe to view the data 
df_feat= pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
```


```python
df_feat.head()
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
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
df_feat.describe()
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
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>...</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14.127292</td>
      <td>19.289649</td>
      <td>91.969033</td>
      <td>654.889104</td>
      <td>0.096360</td>
      <td>0.104341</td>
      <td>0.088799</td>
      <td>0.048919</td>
      <td>0.181162</td>
      <td>0.062798</td>
      <td>...</td>
      <td>16.269190</td>
      <td>25.677223</td>
      <td>107.261213</td>
      <td>880.583128</td>
      <td>0.132369</td>
      <td>0.254265</td>
      <td>0.272188</td>
      <td>0.114606</td>
      <td>0.290076</td>
      <td>0.083946</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.524049</td>
      <td>4.301036</td>
      <td>24.298981</td>
      <td>351.914129</td>
      <td>0.014064</td>
      <td>0.052813</td>
      <td>0.079720</td>
      <td>0.038803</td>
      <td>0.027414</td>
      <td>0.007060</td>
      <td>...</td>
      <td>4.833242</td>
      <td>6.146258</td>
      <td>33.602542</td>
      <td>569.356993</td>
      <td>0.022832</td>
      <td>0.157336</td>
      <td>0.208624</td>
      <td>0.065732</td>
      <td>0.061867</td>
      <td>0.018061</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.981000</td>
      <td>9.710000</td>
      <td>43.790000</td>
      <td>143.500000</td>
      <td>0.052630</td>
      <td>0.019380</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.106000</td>
      <td>0.049960</td>
      <td>...</td>
      <td>7.930000</td>
      <td>12.020000</td>
      <td>50.410000</td>
      <td>185.200000</td>
      <td>0.071170</td>
      <td>0.027290</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.156500</td>
      <td>0.055040</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11.700000</td>
      <td>16.170000</td>
      <td>75.170000</td>
      <td>420.300000</td>
      <td>0.086370</td>
      <td>0.064920</td>
      <td>0.029560</td>
      <td>0.020310</td>
      <td>0.161900</td>
      <td>0.057700</td>
      <td>...</td>
      <td>13.010000</td>
      <td>21.080000</td>
      <td>84.110000</td>
      <td>515.300000</td>
      <td>0.116600</td>
      <td>0.147200</td>
      <td>0.114500</td>
      <td>0.064930</td>
      <td>0.250400</td>
      <td>0.071460</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.370000</td>
      <td>18.840000</td>
      <td>86.240000</td>
      <td>551.100000</td>
      <td>0.095870</td>
      <td>0.092630</td>
      <td>0.061540</td>
      <td>0.033500</td>
      <td>0.179200</td>
      <td>0.061540</td>
      <td>...</td>
      <td>14.970000</td>
      <td>25.410000</td>
      <td>97.660000</td>
      <td>686.500000</td>
      <td>0.131300</td>
      <td>0.211900</td>
      <td>0.226700</td>
      <td>0.099930</td>
      <td>0.282200</td>
      <td>0.080040</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>15.780000</td>
      <td>21.800000</td>
      <td>104.100000</td>
      <td>782.700000</td>
      <td>0.105300</td>
      <td>0.130400</td>
      <td>0.130700</td>
      <td>0.074000</td>
      <td>0.195700</td>
      <td>0.066120</td>
      <td>...</td>
      <td>18.790000</td>
      <td>29.720000</td>
      <td>125.400000</td>
      <td>1084.000000</td>
      <td>0.146000</td>
      <td>0.339100</td>
      <td>0.382900</td>
      <td>0.161400</td>
      <td>0.317900</td>
      <td>0.092080</td>
    </tr>
    <tr>
      <th>max</th>
      <td>28.110000</td>
      <td>39.280000</td>
      <td>188.500000</td>
      <td>2501.000000</td>
      <td>0.163400</td>
      <td>0.345400</td>
      <td>0.426800</td>
      <td>0.201200</td>
      <td>0.304000</td>
      <td>0.097440</td>
      <td>...</td>
      <td>36.040000</td>
      <td>49.540000</td>
      <td>251.200000</td>
      <td>4254.000000</td>
      <td>0.222600</td>
      <td>1.058000</td>
      <td>1.252000</td>
      <td>0.291000</td>
      <td>0.663800</td>
      <td>0.207500</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 30 columns</p>
</div>




```python
#sns.heatmap(df_feat,annot=True)
```


```python
from sklearn.model_selection import train_test_split
```


```python
X = df_feat
y = cancer['target']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=101)
```


```python
# In order to grab the support vector classifier we do the following 
```


```python
from sklearn.svm import SVC
```


```python
model = SVC()
```


```python
model.fit(X_train,y_train)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
predictions = model.predict(X_test)
```


```python
from sklearn.metrics import classification_report,confusion_matrix
```


```python
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

```

    [[  0  83]
     [  0 145]]
    
    
                 precision    recall  f1-score   support
    
              0       0.00      0.00      0.00        83
              1       0.64      1.00      0.78       145
    
    avg / total       0.40      0.64      0.49       228
    


    /Users/bourymbodj/anaconda2/lib/python2.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)



```python
#Our model predicticted that everything belongs to class one 
#The latter means that our model needs to have its values adjusted 
#Through normalization 
#We can search for the best parameters using grid search 
```


```python
from sklearn.grid_search import GridSearchCV
```

    /Users/bourymbodj/anaconda2/lib/python2.7/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    /Users/bourymbodj/anaconda2/lib/python2.7/site-packages/sklearn/grid_search.py:42: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
      DeprecationWarning)



```python
model.fit(X_train,y_train)
# C controls the cost of misclassfication wether its low bias 
# or high variance , The gamma variable is the default kernel 
# A small gamma is the gaussian of a high variance
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
param_grid= {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
```


```python
#Fo rthe following case our estimator is SVC and 
grid = GridSearchCV(SVC(),param_grid, verbose=3)
```


```python
grid.fit(X_train, y_train)
```

    Fitting 3 folds for each of 25 candidates, totalling 75 fits
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ......................... C=0.1, gamma=1, score=0.622807 -   0.0s
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ......................... C=0.1, gamma=1, score=0.622807 -   0.0s
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ......................... C=0.1, gamma=1, score=0.619469 -   0.0s
    [CV] C=0.1, gamma=0.1 ................................................
    [CV] ....................... C=0.1, gamma=0.1, score=0.622807 -   0.0s
    [CV] C=0.1, gamma=0.1 ................................................
    [CV] ....................... C=0.1, gamma=0.1, score=0.622807 -   0.0s
    [CV] C=0.1, gamma=0.1 ................................................
    [CV] ....................... C=0.1, gamma=0.1, score=0.619469 -   0.0s
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] ...................... C=0.1, gamma=0.01, score=0.622807 -   0.0s
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] ...................... C=0.1, gamma=0.01, score=0.622807 -   0.0s
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] ...................... C=0.1, gamma=0.01, score=0.619469 -   0.0s
    [CV] C=0.1, gamma=0.001 ..............................................
    [CV] ..................... C=0.1, gamma=0.001, score=0.622807 -   0.0s
    [CV] C=0.1, gamma=0.001 ..............................................


    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.0s remaining:    0.0s


    [CV] ..................... C=0.1, gamma=0.001, score=0.622807 -   0.0s
    [CV] C=0.1, gamma=0.001 ..............................................
    [CV] ..................... C=0.1, gamma=0.001, score=0.619469 -   0.0s
    [CV] C=0.1, gamma=0.0001 .............................................
    [CV] .................... C=0.1, gamma=0.0001, score=0.912281 -   0.0s
    [CV] C=0.1, gamma=0.0001 .............................................
    [CV] .................... C=0.1, gamma=0.0001, score=0.947368 -   0.0s
    [CV] C=0.1, gamma=0.0001 .............................................
    [CV] .................... C=0.1, gamma=0.0001, score=0.911504 -   0.0s
    [CV] C=1, gamma=1 ....................................................
    [CV] ........................... C=1, gamma=1, score=0.622807 -   0.0s
    [CV] C=1, gamma=1 ....................................................
    [CV] ........................... C=1, gamma=1, score=0.622807 -   0.0s
    [CV] C=1, gamma=1 ....................................................
    [CV] ........................... C=1, gamma=1, score=0.619469 -   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ......................... C=1, gamma=0.1, score=0.622807 -   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ......................... C=1, gamma=0.1, score=0.622807 -   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ......................... C=1, gamma=0.1, score=0.619469 -   0.0s
    [CV] C=1, gamma=0.01 .................................................
    [CV] ........................ C=1, gamma=0.01, score=0.622807 -   0.0s
    [CV] C=1, gamma=0.01 .................................................
    [CV] ........................ C=1, gamma=0.01, score=0.622807 -   0.0s
    [CV] C=1, gamma=0.01 .................................................
    [CV] ........................ C=1, gamma=0.01, score=0.619469 -   0.0s
    [CV] C=1, gamma=0.001 ................................................
    [CV] ....................... C=1, gamma=0.001, score=0.903509 -   0.0s
    [CV] C=1, gamma=0.001 ................................................
    [CV] ....................... C=1, gamma=0.001, score=0.947368 -   0.0s
    [CV] C=1, gamma=0.001 ................................................
    [CV] ....................... C=1, gamma=0.001, score=0.946903 -   0.0s
    [CV] C=1, gamma=0.0001 ...............................................
    [CV] ...................... C=1, gamma=0.0001, score=0.947368 -   0.0s
    [CV] C=1, gamma=0.0001 ...............................................
    [CV] ...................... C=1, gamma=0.0001, score=0.964912 -   0.0s
    [CV] C=1, gamma=0.0001 ...............................................
    [CV] ...................... C=1, gamma=0.0001, score=0.946903 -   0.0s
    [CV] C=10, gamma=1 ...................................................
    [CV] .......................... C=10, gamma=1, score=0.622807 -   0.0s
    [CV] C=10, gamma=1 ...................................................
    [CV] .......................... C=10, gamma=1, score=0.622807 -   0.0s
    [CV] C=10, gamma=1 ...................................................
    [CV] .......................... C=10, gamma=1, score=0.619469 -   0.0s
    [CV] C=10, gamma=0.1 .................................................
    [CV] ........................ C=10, gamma=0.1, score=0.622807 -   0.0s
    [CV] C=10, gamma=0.1 .................................................
    [CV] ........................ C=10, gamma=0.1, score=0.622807 -   0.0s
    [CV] C=10, gamma=0.1 .................................................
    [CV] ........................ C=10, gamma=0.1, score=0.619469 -   0.0s
    [CV] C=10, gamma=0.01 ................................................
    [CV] ....................... C=10, gamma=0.01, score=0.622807 -   0.0s
    [CV] C=10, gamma=0.01 ................................................
    [CV] ....................... C=10, gamma=0.01, score=0.622807 -   0.0s
    [CV] C=10, gamma=0.01 ................................................
    [CV] ....................... C=10, gamma=0.01, score=0.619469 -   0.0s
    [CV] C=10, gamma=0.001 ...............................................
    [CV] ...................... C=10, gamma=0.001, score=0.894737 -   0.0s
    [CV] C=10, gamma=0.001 ...............................................
    [CV] ...................... C=10, gamma=0.001, score=0.938596 -   0.0s
    [CV] C=10, gamma=0.001 ...............................................
    [CV] ...................... C=10, gamma=0.001, score=0.929204 -   0.0s
    [CV] C=10, gamma=0.0001 ..............................................
    [CV] ..................... C=10, gamma=0.0001, score=0.938596 -   0.0s
    [CV] C=10, gamma=0.0001 ..............................................
    [CV] ..................... C=10, gamma=0.0001, score=0.947368 -   0.0s
    [CV] C=10, gamma=0.0001 ..............................................
    [CV] ..................... C=10, gamma=0.0001, score=0.946903 -   0.0s
    [CV] C=100, gamma=1 ..................................................
    [CV] ......................... C=100, gamma=1, score=0.622807 -   0.0s
    [CV] C=100, gamma=1 ..................................................
    [CV] ......................... C=100, gamma=1, score=0.622807 -   0.0s
    [CV] C=100, gamma=1 ..................................................
    [CV] ......................... C=100, gamma=1, score=0.619469 -   0.0s
    [CV] C=100, gamma=0.1 ................................................
    [CV] ....................... C=100, gamma=0.1, score=0.622807 -   0.0s
    [CV] C=100, gamma=0.1 ................................................
    [CV] ....................... C=100, gamma=0.1, score=0.622807 -   0.0s
    [CV] C=100, gamma=0.1 ................................................
    [CV] ....................... C=100, gamma=0.1, score=0.619469 -   0.0s
    [CV] C=100, gamma=0.01 ...............................................
    [CV] ...................... C=100, gamma=0.01, score=0.622807 -   0.0s
    [CV] C=100, gamma=0.01 ...............................................
    [CV] ...................... C=100, gamma=0.01, score=0.622807 -   0.0s
    [CV] C=100, gamma=0.01 ...............................................
    [CV] ...................... C=100, gamma=0.01, score=0.619469 -   0.0s
    [CV] C=100, gamma=0.001 ..............................................
    [CV] ..................... C=100, gamma=0.001, score=0.894737 -   0.0s
    [CV] C=100, gamma=0.001 ..............................................
    [CV] ..................... C=100, gamma=0.001, score=0.938596 -   0.0s
    [CV] C=100, gamma=0.001 ..............................................
    [CV] ..................... C=100, gamma=0.001, score=0.929204 -   0.0s
    [CV] C=100, gamma=0.0001 .............................................
    [CV] .................... C=100, gamma=0.0001, score=0.929825 -   0.0s
    [CV] C=100, gamma=0.0001 .............................................
    [CV] .................... C=100, gamma=0.0001, score=0.947368 -   0.0s
    [CV] C=100, gamma=0.0001 .............................................
    [CV] .................... C=100, gamma=0.0001, score=0.920354 -   0.0s
    [CV] C=1000, gamma=1 .................................................
    [CV] ........................ C=1000, gamma=1, score=0.622807 -   0.0s
    [CV] C=1000, gamma=1 .................................................
    [CV] ........................ C=1000, gamma=1, score=0.622807 -   0.0s
    [CV] C=1000, gamma=1 .................................................
    [CV] ........................ C=1000, gamma=1, score=0.619469 -   0.0s
    [CV] C=1000, gamma=0.1 ...............................................
    [CV] ...................... C=1000, gamma=0.1, score=0.622807 -   0.0s
    [CV] C=1000, gamma=0.1 ...............................................
    [CV] ...................... C=1000, gamma=0.1, score=0.622807 -   0.0s
    [CV] C=1000, gamma=0.1 ...............................................
    [CV] ...................... C=1000, gamma=0.1, score=0.619469 -   0.0s
    [CV] C=1000, gamma=0.01 ..............................................
    [CV] ..................... C=1000, gamma=0.01, score=0.622807 -   0.0s
    [CV] C=1000, gamma=0.01 ..............................................
    [CV] ..................... C=1000, gamma=0.01, score=0.622807 -   0.0s
    [CV] C=1000, gamma=0.01 ..............................................
    [CV] ..................... C=1000, gamma=0.01, score=0.619469 -   0.0s
    [CV] C=1000, gamma=0.001 .............................................
    [CV] .................... C=1000, gamma=0.001, score=0.894737 -   0.0s
    [CV] C=1000, gamma=0.001 .............................................
    [CV] .................... C=1000, gamma=0.001, score=0.938596 -   0.0s
    [CV] C=1000, gamma=0.001 .............................................
    [CV] .................... C=1000, gamma=0.001, score=0.929204 -   0.0s
    [CV] C=1000, gamma=0.0001 ............................................
    [CV] ................... C=1000, gamma=0.0001, score=0.903509 -   0.0s
    [CV] C=1000, gamma=0.0001 ............................................
    [CV] ................... C=1000, gamma=0.0001, score=0.947368 -   0.0s
    [CV] C=1000, gamma=0.0001 ............................................
    [CV] ................... C=1000, gamma=0.0001, score=0.929204 -   0.0s


    [Parallel(n_jobs=1)]: Done  75 out of  75 | elapsed:    1.2s finished





    GridSearchCV(cv=None, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]},
           pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=3)




```python
#Continue later 
```
