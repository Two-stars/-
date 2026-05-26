# All Features Dataset Split Report

## 数据来源
- Excel 路径: 流体识别数据.xlsx
- Sheet: Sheet1

## 参考划分
- train.csv: dataset\train.csv
- val.csv: dataset\val.csv
- test.csv: dataset\test.csv

## 输出文件
- dataset\all_features\train.csv
- dataset\all_features\val.csv
- dataset\all_features\test.csv

## 样本数量
- train: 原 1088 vs all_features 1088
- val: 原 243 vs all_features 243
- test: 原 253 vs all_features 253

## 井数量
- train wells 数量: 52
- val wells 数量: 13
- test wells 数量: 13

## 输入特征数量
- 24

## 输入特征列表
- GR
- SP
- AC
- AC1
- CAL
- U
- DEN
- DEN1
- CNL
- CNL1
- PE
- RD
- RS
- log（RD）
- log（RS）
- m2r3
- m2r6
- m2r9
- POR
- SW
- PERM
- ∆ Φ1
- ∆ Φ2
- ∆ Φ3

## 说明
- 本数据集从原始 Excel Sheet1 生成；
- 使用已有 train/val/test 的井划分；
- 未重新划分；
- 未合并 train/val/test；
- 未新增 Excel 中不存在的特征列；
- 保留 Excel 中已有全部原始列；
- 井名、深度、试油结论、y_fine、y_coarse 保留在 CSV 中，但不作为模型输入。
