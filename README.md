# RAG知识库问答系统

基于 LangChain + Chroma + DeepSeek 构建的企业内部知识库问答系统。

## 功能特性

- 支持 txt / pdf 多格式文档加载
- 支持多文件知识库构建
- 基于向量相似度的智能检索
- 多轮对话，理解上下文指代
- 自动标注答案来源文件
- 知识库本地持久化

## 技术栈

- LangChain：LLM应用开发框架
- Chroma：本地向量数据库
- BAAI/bge-m3：中文Embedding模型
- DeepSeek：大语言模型
- Python 3.11

## 快速开始

1. 安装依赖
pip install -r requirements.txt

2. 配置环境变量
复制 .env.example 为 .env，填入API Key

3. 把文档放入docs文件夹

4. 运行
python rag_chatbot.py

## 项目结构
```
├── docs/              # 知识库文档
├── rag_db/            # 向量数据库（自动生成）
├── rag_chatbot_01.py     # 主程序第一版
├── rag_chatbot_02.py     # 主程序第二版
├── rag_chatbot_03.py     # 主程序第三版
├── .env               # API Key配置（不上传）
└── README.md
```
