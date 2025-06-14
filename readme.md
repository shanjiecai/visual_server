# 视频流处理服务

一个高性能、高扩展性的视频流处理框架，基于异步架构设计，支持实时视频分析、大模型处理和智能对话。

## 🌟 核心特性

- **异步架构**: 基于asyncio的高性能异步处理
- **模块化设计**: 面向接口编程，支持插件化扩展
- **多种视频源**: 支持文件、摄像头、RTMP流、WebRTC等多种视频输入
- **智能预处理**: 内置YOLO检测、相似帧过滤等预处理器
- **大模型集成**: 支持OpenAI、自定义大模型等多种AI服务
- **消息队列**: 支持Kafka等消息队列系统
- **后处理器**: 对话发起、通知发送等智能后处理

## 🎯 设计理念

### 高效过滤机制
框架采用多级过滤机制减少计算资源浪费：

1. **时间过滤**: 按秒取帧而非处理所有帧
2. **相似度过滤**: 自动过滤相似度高的连续帧
3. **目标过滤**: 基于检测结果过滤，只处理包含特定目标的帧
4. **置信度过滤**: 根据检测置信度决定是否进行进一步处理

### 单队列多任务
- **统一消息格式**: 所有视觉任务共用同一消息格式
- **任务类型识别**: 基于预处理结果自动确定最适合的任务类型
- **并行处理**: 单个队列支持多个Worker并行消费

### 大模型高效利用
- **前置过滤**: 只有真正需要分析的帧才会发送到大模型
- **异步处理**: 支持异步调用，避免阻塞主处理流程
- **结果解析**: 统一的结果解析机制

## 📁 项目结构

```
visual_server/
├── core/                    # 核心模块
│   ├── interfaces.py        # 抽象接口定义
│   ├── factory.py          # 工厂模式实现
│   ├── filters.py          # 过滤器实现
│   ├── pipeline.py         # 流水线处理框架
│   └── service_manager.py   # 服务生命周期管理
├── producer/                # 视频源生产者
│   ├── base.py             # 视频源基类
│   ├── camera_source.py    # 摄像头视频源
│   ├── file_source.py      # 文件视频源
│   ├── rtmp_source.py      # RTMP流视频源
│   └── webrtc_source.py    # WebRTC视频源
├── preprocessor/            # 预处理模块
│   ├── base.py             # 预处理器基类
│   ├── yolo_detector.py    # YOLO检测器
│   ├── similar_frame_filter.py # 相似帧过滤器
│   └── memory_extractor.py # 记忆提取器
├── message_queue/          # 消息队列模块
│   ├── base.py             # 队列基类
│   ├── memory_queue.py     # 内存队列
│   └── kafka_queue.py      # Kafka队列
├── consumer/                # 消费者模块
│   ├── base.py             # 大模型处理器基类
│   └── openai_vlm.py       # OpenAI视觉大模型处理器
├── worker/                  # 工作进程
│   ├── vlm_worker.py       # 视觉大模型工作进程
│   └── vlm_worker_config.yaml # 工作进程配置
├── postprocessor/           # 后处理模块
│   ├── base.py             # 后处理器基类
│   └── dialogue_initiator.py # 对话发起器
├── utils/                   # 工具模块
│   ├── cache.py            # 缓存实现
│   └── config.py           # 配置管理
├── demo.py                  # 主要演示程序
├── config.yaml             # 主配置文件
└── requirements.txt        # 依赖清单
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 推荐使用虚拟环境

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd visual_server

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 基础配置

编辑配置文件 `config.yaml`：

```yaml
# 视频源配置
camera:
  camera_index: 0          # 摄像头索引
  fps: 1.0                # 每秒取帧数
  resolution: [640, 480]   # 分辨率

# 预处理器配置
preprocessors:
  similar_frame_filter:
    enabled: true
    similarity_threshold: 0.9
  
  yolo_detector:
    enabled: true
    model_path: "models/yolo-v8l-seg.pt"
    target_classes: ["person"]

# 队列配置
queue:
  bootstrap_servers: ["localhost:9092"]
  topic_name: "video_processing"
  use_kafka: true
```

### 运行服务

有多种方式启动系统：

#### 方式1：使用主入口文件（推荐）
```bash
# 只运行演示程序（摄像头处理）
python main.py demo

# 只运行VLM工作进程
python main.py worker

# 同时运行两者
python main.py both

# 使用自定义配置
python main.py demo --config my_config.yaml
```

#### 方式2：分别启动
```bash
# 启动主处理程序
python demo.py

# 启动视觉大模型工作进程
python -m worker.vlm_worker --config worker/vlm_worker_config.yaml
```

#### 方式3：使用进程管理器
```bash
# 启动演示程序
python run.py start demo

# 启动VLM工作进程
python run.py start worker

# 查看状态
python run.py status

# 停止所有进程
python run.py stop
```

## 🔧 支持的视频源

### 摄像头
```yaml
video_source:
  source_type: "camera"
  config:
    camera_index: 0
    fps: 1.0
    resolution: [640, 480]
```

### 视频文件
```yaml
video_source:
  source_type: "file"
  config:
    file_path: "path/to/video.mp4"
    fps: 1.0
```

### RTMP流
```yaml
video_source:
  source_type: "rtmp"
  config:
    url: "rtmp://example.com/live/stream"
    fps: 1.0
```

### WebRTC流
```yaml
video_source:
  source_type: "webrtc"
  config:
    url: "http://server.com/api/whep/stream"
    max_frames_buffer: 3
    connection_timeout: 10
```

## ⚙️ 核心配置

### 相似帧过滤器
```yaml
similar_frame_filter:
  similarity_threshold: 0.9    # 相似度阈值
  comparison_method: "histogram" # 比较方法
  history_size: 5             # 历史帧数量
  min_time_interval: 0.5      # 最小时间间隔
```

### YOLO检测器
```yaml
yolo_detector:
  model_path: "models/yolo-v8l-seg.pt"
  device: "cpu"               # 计算设备
  confidence_threshold: 0.5   # 置信度阈值
  target_classes: ["person"]  # 目标类别
```

### VLM工作进程
```yaml
vlm_config:
  base_url: "https://api.openai.com/v1"
  api_key: "your_api_key"
  model_name: "gpt-4-vision-preview"
  max_tokens: 128
  temperature: 0.7
```

## 🔌 扩展开发

### 自定义视频源

```python
from producer.base import BaseVideoSource
from core.interfaces import FrameData

class CustomVideoSource(BaseVideoSource):
    async def _do_initialize(self) -> bool:
        # 初始化自定义视频源
        return True
    
    async def _get_next_frame(self) -> FrameData:
        # 获取下一帧
        return FrameData(...)
    
    async def _do_close(self) -> None:
        # 清理资源
        pass
```

### 自定义预处理器

```python
from preprocessor.base import BasePreprocessor
from core.interfaces import ProcessingResult

class CustomPreprocessor(BasePreprocessor):
    @property
    def processor_name(self) -> str:
        return "custom_processor"
    
    async def _do_process(self, frame_data: FrameData) -> ProcessingResult:
        # 实现自定义处理逻辑
        return ProcessingResult(...)
```

### 自定义后处理器

```python
from postprocessor.base import BasePostprocessor

class CustomPostprocessor(BasePostprocessor):
    @property
    def processor_name(self) -> str:
        return "custom_postprocessor"
    
    async def _do_execute(self, task: ProcessingTask) -> Dict[str, Any]:
        # 实现自定义后处理逻辑
        return {"status": "completed"}
```