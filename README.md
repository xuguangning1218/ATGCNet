# EATG-Net
Official code for paper 《ATGCNet: An Adaptive Tensor Graph Convolution Network for Air Quality Prediction》

### Overall Architecture of ATGCNet-Net
![image](https://github.com/xuguangning1218/ATGCNet/blob/master/figure/model.png)

### Source Files Description

```
-- data # dataset folder
  -- sample_match_DPT.csv # samples of DPT
  -- sample_match_PRS.csv # samples of PRS
  -- sample_match_RHU.csv # samples of RHU
  -- sample_match_TEM.csv # samples of TEM
  -- sample_match_WIN_D_INST.csv # samples of WIN_D_INST
  -- sample_match_WIN_S_INST.csv # samples of WIN_S_INST
  -- sample_PM10.csv # samples of PM10
  -- sample_PM10_masker.csv # samples of PM10 makser
  -- sample_PM25.csv # samples of PM2.5
  -- sample_PM25_masker.csv # samples of PM2.5 of PM10 makser
-- dataprovider # data provider folder
  -- dataprovider.py # necessary data process to form train, validate, test loader. 
-- model # necessary layer
  -- ATGCO.py # the proposed ATGCO layer for both slice and tensor approximation
  -- ATG_GRU.py #  the proposed ATG-GRU layer
  -- ATGCNet.py # the proposed ATGCNet model
  -- network.py # the network handler for model train, validate, test, save, and load
-- utility # utility tool
  -- metrics.py # the metrics function
ATGCNet.config # the hyper, model, and train parameters for ATGCNet
ATGCNet.py # main running code for the ATGCNet
```
