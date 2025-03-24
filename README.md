# flick-pick

## Shared data
path: google drive - /5003-BigData/data

download or use it in colab:
```
from google.colab import drive
drive.mount('/content/drive')
!ls -1 /content/drive/MyDrive/5003-BigData/data
```



## Install Issue

- If there is error `No module named 'distutils'`，when python >= v3.12, 
run 
	`pip install -U pip setuptools wheel`
or 
	`pip install packaging`

- local Spark dependents on `openjdk@17`