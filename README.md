# MF-Former

基于 PyTorch 的 **MF-Former** 实现，用于 WeatherBench 数据集上的短时降水预测任务。

## 简介

MF-Former 是一个面向短时降水预测的多模态融合模型。  
模型同时利用降水序列和辅助气象变量，通过双流编码、多尺度特征融合和 U 型解码结构，提高降水预测精度。

## 数据集

本项目使用 **WeatherBench** 数据集，输入变量包括：

- `tp`：total precipitation
- `rh`：850 hPa relative humidity
- `t`：850 hPa temperature
- `u`：850 hPa u wind
- `v`：850 hPa v wind

实验设置如下：

- 输入长度：6 小时
- 预测长度：6 小时
- 训练集：2014–2016
- 验证集：2017
- 测试集：2018

## 文件结构

```text
MF-Former/
├── datasets/
│   ├── weatherbench.py
│   └── __init__.py
├── models/
│   ├── mf_former.py
│   ├── encoder.py
│   ├── MS_UNet.py
│   └── __init__.py
├── utils/
│   ├── initial.py
│   ├── evaluation.py
│   └── __init__.py
├── LICENSE
├── README.md
├── configs.py
├── train.py
└── test.py
```

## 数据组织方式

请将处理后的数据放在 `./data` 目录下，示例如下：

```text
data/
├── tp/
│   ├── total_precipitation_2014_2016_1.40625deg.nc
│   ├── total_precipitation_2017_1.40625deg.nc
│   └── total_precipitation_2018_1.40625deg.nc
├── rh/
│   ├── relative_humidity_850_2014_2016_1.40625deg.nc
│   ├── relative_humidity_850_2017_1.40625deg.nc
│   └── relative_humidity_850_2018_1.40625deg.nc
├── t/
│   ├── temperature_850_2014_2016_1.40625deg.nc
│   ├── temperature_850_2017_1.40625deg.nc
│   └── temperature_850_2018_1.40625deg.nc
├── u/
│   ├── u_850_component_of_wind_2014_2016_1.40625deg.nc
│   ├── u_850_component_of_wind_2017_1.40625deg.nc
│   └── u_850_component_of_wind_2018_1.40625deg.nc
└── v/
    ├── v_850_component_of_wind_2014_2016_1.40625deg.nc
    ├── v_850_component_of_wind_2017_1.40625deg.nc
    └── v_850_component_of_wind_2018_1.40625deg.nc
```

## 环境依赖

建议环境：

- Python 3.9+
- PyTorch
- NumPy
- xarray
- netCDF4
- scikit-image

安装主要依赖：

```bash
pip install torch numpy xarray netCDF4 scikit-image
```

## 训练

```bash
python train.py
```

训练配置见 `configs.py`。

## 测试

```bash
python test.py
```

默认测试权重路径为：

```text
./saved_models/mf_former.pth
```

## 评估指标

本项目使用以下指标评估模型性能：

- RMSE
- MAE
- POD
- FAR
- CSI

## 说明

- 代码面向 WeatherBench 数据集上的短时降水预测实验。
- 训练过程中会自动保存模型权重。
- 使用前请确认 `configs.py` 中的数据路径设置正确。

## 引用

如果本项目对你的研究有帮助，请引用对应论文。

## License

MIT License
