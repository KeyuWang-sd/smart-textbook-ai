<p align="center">
  <img src="docs/banner.svg" alt="SmartTextbook AI Banner" width="800" />
</p>

<p align="center">
  <b>If you can ask, it can teach.</b><br>
  <i>面向 K12 教育的 RAG 智能问答系统 — 从教材到答案，一步到位</i>
</p>

---

<p align="center">
  <table align="center">
    <tr>
      <td align="center" width="200">
        <img src="https://img.icons8.com/fluency/48/chatbot.png" width="40"/><br>
        <b>智能问答</b><br>
        <sub>四维过滤检索</sub>
      </td>
      <td align="center" width="200">
        <img src="https://img.icons8.com/fluency/48/test-passed.png" width="40"/><br>
        <b>试题生成</b><br>
        <sub>5种题型 × 3级难度</sub>
      </td>
      <td align="center" width="200">
        <img src="https://img.icons8.com/fluency/48/combo-chart.png" width="40"/><br>
        <b>错题分析</b><br>
        <sub>LLM 知识点推理</sub>
      </td>
      <td align="center" width="200">
        <img src="https://img.icons8.com/fluency/48/folder-invoices.png" width="40"/><br>
        <b>文档管理</b><br>
        <sub>12 年级全册覆盖</sub>
      </td>
    </tr>
  </table>
</p>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    🌐 用户浏览器                           │
│              http://localhost:3000                       │
│         HTML5 + CSS3 + KaTeX 公式渲染                     │
└──────────────────────┬──────────────────────────────────┘
                       │  REST API
                       ▼
┌─────────────────────────────────────────────────────────┐
│              ⚡ FastAPI 后端 (:8000)                      │
│                                                        │
│   ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│   │ 文档加载  │  │ RAG 检索  │  │  LLM 智能推理     │    │
│   │ 自动检测  │  │ 四维过滤  │  │  DeepSeek V4     │    │
│   │ 年级/学科 │  │ 阈值 0.5 │  │  问答+出题+分析   │    │
│   └──────────┘  └──────────┘  └──────────────────┘    │
│                                                        │
│   ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│   │ pgvector │  │  Redis   │  │  百炼 Embedding    │    │
│   │ 向量存储  │  │ 查询缓存  │  │  text-embed-v3   │    │
│   └──────────┘  └──────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Why SmartTextbook AI?

| 🎯 精准检索 | 🧒 难度自适应 | 🧠 智能分析 | 🎨 黑板风 UI |
|:---:|:---:|:---:|:---:|
| 年级 + 学科 + 章节 + 知识点<br>四维过滤，只找对的内容 | 根据学生年级自动调整<br>用语难度和解释深度 | 错题→LLM 推理→教材知识点<br>自动匹配所属章节 | 原生 HTML/CSS<br>KaTeX 数学公式完美渲染 |

---

## 🚀 Quick Start

```bash
# 1️⃣ 启动数据库（PostgreSQL + pgvector + Redis）
docker compose up -d postgres redis

# 2️⃣ 安装依赖
pip install -r requirements.txt

# 3️⃣ 启动后端 API
python -m app.edu_api

# 4️⃣ 启动前端（另一个终端）
cd ui && python -m http.server 3000
```

然后打开 **http://localhost:3000** 🎉

<details>
<summary>📦 首次使用？一键导入教材文档</summary>

上传文档后，通过 API 一键入库（支持 PDF/DOCX/MD/TXT）：

```bash
# 方式 1：通过 API 上传 + 导入
curl -X POST "http://localhost:8000/api/upload?doc_type=textbook&grade=初三&subject=数学" \
  -F "file=@data/textbooks/初三数学.md"
curl -X POST "http://localhost:8000/api/ingest" \
  -H "Content-Type: application/json" \
  -d '{"doc_type":"textbook","grade":"初三","subject":"数学"}'

# 方式 2：批量导入所有教材（在 app.edu_api 运行状态下）
python -c "
from app.edu_document_loader import load_from_directory
from app.edu_splitter import split_documents
from app.edu_rag_engine import EduVectorStore
from app.config import settings

for grade in ['初一','初二','初三']:
    for subject in ['数学']:
        docs = load_from_directory('data/textbooks', doc_type='textbook', grade=grade, subject=subject)
        chunks = split_documents(docs)
        store = EduVectorStore(settings.database_url)
        store.create_collection(chunks, grade=grade, subject=subject)
        print(f'✓ {grade} {subject} → {len(chunks)} chunks')
```
</details>

---

## 🛠️ Tech Stack

| Layer | Technology |
|:---|:---|
| 🎨 Frontend | HTML5 · CSS3 · JavaScript · KaTeX |
| ⚡ Backend | FastAPI · Uvicorn · LangChain |
| 🧠 LLM | DeepSeek V4 Flash |
| 🧬 Embedding | 百炼 text-embedding-v3（1024 dims） |
| 📊 Reranker | 百炼 gte-rerank-v2（API，无需本地模型） |
| 🗄️ Vector DB | PostgreSQL + pgvector |
| 🔴 Cache | Redis |
| 📄 Parsing | PyPDF · python-docx · Unstructured |
| 🐳 DevOps | Docker · Docker Compose |

---

## 📂 Project Structure

```
rag_test/
├── app/                          # 后端核心
│   ├── config.py                 # 配置管理 (.env)
│   ├── edu_document_loader.py    # 文档加载 & 自动检测
│   ├── edu_splitter.py           # 智能切分 & 章节识别
│   ├── edu_rag_engine.py         # RAG 引擎 (检索/重排/生成)
│   ├── edu_features.py           # 试题生成/图谱/错题分析
│   └── edu_api.py                # 8 个 RESTful 接口
├── ui/
│   ├── index.html                # 黑板风格前端 (2647 行)
│   └── nginx-static.conf         # Nginx 静态服务配置
├── data/textbooks/               # 1~12 年级数学全册 📚
├── .dockerignore                 # Docker 安全加固
├── .env.example                  # 环境变量模板
├── docker-compose.yml            # 容器编排
└── requirements.txt              # 依赖清单
```

---

## 🎯 Learning Path

```
📖 上传教材 ──→ 🔬 向量化入库 ──→ 💬 智能问答
                    │
                    ├── 📝 试题生成 (5 种题型)
                    ├── 📊 错题分析 (LLM 知识点映射)
                    └── 📁 文档管理 (上传/删除)
```

---

## 🌍 你好 · Hello · こんにちは · Bonjour

本项目支持 1~12 年级数学全册教材，覆盖小学到高中全部知识点。

欢迎世界各地的教育者和开发者一起贡献！🌟

---

<p align="center">
  <sub>Built with ❤️ using FastAPI + LangChain + DeepSeek + pgvector</sub>
</p>
