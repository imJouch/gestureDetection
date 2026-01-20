# 手语识别系统 | Gesture Detection System

一个基于深度学习的手语识别和翻译系统，支持对连续手语视频的实时识别和句子级别的翻译。

## 功能特性

- 🎥 支持视频格式的手语识别
- 🧠 采用多种深度学习模型架构（3D ResNet、Seq2Seq、GCN等）
- 📝 支持连续手语的句子级翻译
- ⚡ GPU加速支持（CUDA）
- 🎯 支持500+种手语词汇
- 🔄 灵活的批处理和数据预处理

## 系统要求

- Python 3.7+
- PyTorch 1.9+
- CUDA 11.0+ (可选，用于GPU加速)
- 其他依赖项详见 `requirements.txt`

## 项目结构

```
gestureDetection/
├── src/                    # 源代码目录
│   ├── detect.py          # 检测脚本
│   ├── train.py           # 模型训练脚本
│   ├── dataset.py         # 数据集定义
│   ├── cutFrame.py        # 视频帧处理
│   └── tools.py           # 工具函数
├── models/                # 模型架构
│   ├── Seq2Seq.py        # 序列到序列模型
│   ├── Conv3D.py         # 3D卷积模型
│   ├── ConvLSTM.py       # ConvLSTM模型
│   ├── RNN.py            # RNN模型
│   ├── GCN.py            # 图卷积网络
│   └── Attention.py      # 注意力机制
├── trainedModel/         # 预训练模型
├── video/                # 输入视频文件夹
├── picture/              # 处理后的帧图片
├── label/                # 标签和字典
└── LICENSE

```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将要识别的手语视频放入 `video/` 文件夹中。

### 3. 运行识别

```bash
cd src
python detect.py --data_path ../picture \
                  --label_path ../label/dictionary.txt \
                  --model_path ../trainedModel/gestureDetection.pth
```

### 配置参数

运行以下命令查看所有可用参数：

```bash
python detect.py --help
```

主要参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | `../picture` | 测试数据路径 |
| `--label_path` | `../label/dictionary.txt` | 标签文件路径 |
| `--model` | `3dresnet18` | 选择使用的模型 |
| `--model_path` | `../trainedModel/gestureDetection.pth` | 模型权重文件路径 |
| `--num_classes` | `500` | 手语词汇数量 |
| `--batch_size` | `2` | 批处理大小 |
| `--sample_size` | `128` | 采样尺寸 |
| `--sample_duration` | `48` | 采样持续帧数 |
| `--cuda_devices` | `0` | GPU设备ID |
| `--no_cuda` | - | 不使用GPU |

## 支持的模型

本项目支持多种模型架构：

- **3D ResNet** - 3D卷积残差网络
- **Seq2Seq** - 序列到序列模型，支持编码器-解码器架构
- **ConvLSTM** - 卷积LSTM模型
- **GCN** - 图卷积网络
- **RNN** - 循环神经网络
- **Attention** - 注意力机制模型

## 训练模型

如需在自己的数据集上训练模型，运行：

```bash
cd src
python train.py [training_parameters]
```

## 输出格式

系统将输出：
- 识别的手语词汇序列
- 翻译后的完整句子
- 每帧的识别置信度

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 作者

Josh Wang

## 贡献

欢迎提交 Issue 和 Pull Request！

## 相关资源

- [PyTorch 文档](https://pytorch.org/docs/)
- [手语识别相关论文]()

## 常见问题

**Q: 如何处理自己的手语视频？**
A: 将视频文件放入 `video/` 目录，运行 `detect.py` 脚本即可。系统会自动进行帧提取和预处理。

**Q: 需要GPU吗？**
A: 不是必须的，但使用GPU会显著提升处理速度。可以使用 `--no_cuda` 参数在CPU模式下运行。

**Q: 支持哪些视频格式？**
A: 支持OpenCV支持的格式，包括 .mp4, .avi, .mov 等常见格式。
