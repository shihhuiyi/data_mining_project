執行run.py主程式
特徵值的重要性放在feature資料夾->最後會用重要性非0的特徵值
預測結果圖片存在plot資料夾
configs.json可以修改訓練參數

流程：
1.資料前處理
1-1資料過去
刪除無用的特徵值，如起造人、起造編號等等...
刪除具有空值的資料
刪除極端值(1.5IQR)
1-2資料正規化
使用minmax(0,1)進行正規化

2.特徵選取
使用xgboost進行特徵選取
會將特徵值重要性大於0的特徵放入後續模型進行訓練

3.模型訓練
xgboost
random forest
knn
logistic regression