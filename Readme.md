執行run.py主程式
<br>特徵值的重要性放在feature資料夾->最後會用重要性非0的特徵值
<br>預測結果圖片存在plot資料夾
<br>configs.json可以修改訓練參數
<br>
<br>流程：
<br>1.資料前處理
<br>1-1資料過去
<br>刪除無用的特徵值，如起造人、起造編號等等...
<br>刪除具有空值的資料
<br>刪除極端值(1.5IQR)
<br>1-2資料正規化
<br>使用minmax(0,1)進行正規化
<br>
<br>2.特徵選取
<br>使用xgboost進行特徵選取
<br>會將特徵值重要性大於0的特徵放入後續模型進行訓練
<br>
<br>3.模型訓練
<br>xgboost
<br>random forest
<br>knn
<br>logistic regression
<br>
<br>requirements:
<br>python 3.7
<br>sklearn
<br>numpy
<br>pandas
<br>xgboost
<br>matplotlib
