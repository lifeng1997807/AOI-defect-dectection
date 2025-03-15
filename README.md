# AOI-defect detection
AOI  defect dectection by vgg16

摘要:
透過vgg16 對圖片做出分類，其中自訂義了損失函數並做出混淆矩陣，並嘗試運用不同圖像處理方法但效果不佳，要進一步提高精度可能要從模型下手

data resource
https://aidea-web.tw/playground

performance
acc=0.9893598,rank:171/835

reference
https://github.com/jellyfish1456/AOI-defect-detection

#更:我還做了凍結預訓練權重/利用svm取代softmax 輸出並導入難樣本挖掘<但效果和上面的差不多
