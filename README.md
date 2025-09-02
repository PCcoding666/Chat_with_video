# MiniCPM-V 视频聊天 Demo - Intel GPU版本

🎥 一个使用 MiniCPM-V-4.5 模型在 Intel Arc GPU 上进行视频理解和对话的演示程序。

## 特性

- 🚀 **Intel Arc GPU 加速**: 使用 Intel XPU 进行高效推理
- 🎥 **多种视频格式**: 支持 MP4, AVI, MOV, MKV 等常见格式
- 🤖 **智能对话**: 基于 MiniCPM-V-4.5 多模态大语言模型
- 💬 **中英文支持**: 支持中英文视频内容理解和对话
- 📊 **批量处理**: 支持一次处理多个问题
- 🛠️ **交互式界面**: 友好的命令行交互界面

## 系统要求

### 硬件要求
- Intel Arc GPU (130V 或更高版本)
- 16GB+ 系统内存
- 10GB+ 可用存储空间

### 软件要求
- Windows 11 / Ubuntu 24.04+
- Python 3.11+
- Intel GPU 驱动程序
- uv (包管理器)

## 安装指南

### 1. 克隆仓库
```bash
git clone https://github.com/PCcoding666/Chat_with_video.git
cd Chat_with_video
```

### 2. 安装依赖
```bash
# 使用 uv 初始化项目
uv sync

# 或者使用 pip 安装
pip install -r requirements.txt

# 安装 Intel XPU 版本的 PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

### 3. 验证安装
```bash
# 运行系统测试
uv run python main.py --test
```

## 使用方法

### 交互式模式
```bash
# 启动交互式聊天
uv run python main.py

# 指定视频文件
uv run python main.py --video /path/to/your/video.mp4
```

### 批量处理模式
```bash
# 启动批量处理
uv run python main.py --batch
```

### 系统测试
```bash
# 运行系统测试
uv run python main.py --test
```

## 示例用法

1. **视频内容理解**:
   - “请描述这个视频的内容”
   - “What is happening in this video?”

2. **细节提取**:
   - “视频中有多少人？”
   - “背景音乐是什么风格？”

3. **情绪分析**:
   - “视频中的人物情绪如何？”
   - “这个场景给你什么感觉？”

## 项目结构

```
Chat_with_video/
├── main.py                    # 主程序入口
├── video_encoder.py           # 视频编码模块
├── model_loader.py            # 模型加载器
├── video_chat_interface.py    # 视频聊天接口
├── pyproject.toml             # 项目配置
├── requirements.txt           # 依赖列表
└── README.md                  # 项目说明
```

## 技术架构

- **模型**: MiniCPM-V-4.5 (多模态大语言模型)
- **推理引擎**: Intel XPU (Intel Arc GPU)
- **视频处理**: decord + PIL + OpenCV
- **数值计算**: NumPy + SciPy
- **模型库**: Transformers

## 性能优化

- 使用 bfloat16 精度减少显存占用
- SDPA 注意力机制优化
- 动态视频帧采样策略
- 3D 重采样数据压缩
- Intel XPU 内存管理优化

## 常见问题

### Q: 如何检查 Intel GPU 是否可用？
A: 运行 `python -c "import torch; print(torch.xpu.is_available())"`

### Q: 模型下载失败怎么办？
A: 请检查网络连接，或者手动下载模型到本地。

### Q: 显存不够怎么办？
A: 可以尝试降低 `choose_fps` 参数或使用更小的视频分辨率。

## 贡献

欢迎提交 Issues 和 Pull Requests！

## 许可证

MIT License

## 致谢

- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) - 多模态大语言模型
- [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) - Intel GPU 支持
- [decord](https://github.com/dmlc/decord) - 视频解码库