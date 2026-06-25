# edu_api.py
"""智能教材问答与学习辅助平台 —— FastAPI 后端

启动方式:
    python -m app.edu_api
    # 或
    uvicorn app.edu_api:app --host 0.0.0.0 --port 8000 --reload

API 接口:
    POST /api/upload          - 上传教育文档
    POST /api/chat            - 教育问答（支持 session_id 记忆）
    POST /api/agent           - 🆕 统一 Agent 接口（意图识别+多步编排+记忆）
    POST /api/exam/generate   - 试题生成
    POST /api/image/analyze   - 🆕 图片内容识别（OCR）
    POST /api/image/solve     - 🆕 拍照解题
    GET  /api/knowledge/graph - 知识点图谱
    POST /api/study/analyze   - 错题分析
    GET  /health              - 健康检查
"""

import os
import re
import sys
import json
import uuid
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings  # noqa: E402


def _safe_filename(filename: str) -> str:
    """防路径遍历：剥离目录，只保留安全文件名"""
    # 只取最后一个路径分隔符后的部分（防止 ../../etc/passwd）
    name = os.path.basename(filename)
    # 移除所有非安全字符，只保留中英文、数字、._-
    safe = re.sub(r'[^\w一-鿿.\-_]', '_', name)
    # 防止空文件名
    if not safe or safe.startswith('.'):
        safe = "file_" + safe
    # 加随机前缀防冲突
    stem, ext = os.path.splitext(safe)
    return "{}_{}{}".format(stem, uuid.uuid4().hex[:8], ext)


# ================================================================
# FastAPI 应用
# ================================================================

app = FastAPI(
    title="🎓 智能教材问答与学习辅助平台",
    description="面向教育领域的 RAG 问答系统，支持教材/课件/题库的智能检索与学习辅助",
    version="1.0.0",
)

# CORS（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# 全局状态（简化版，生产环境应使用连接池）
# ================================================================

# 存储已上传文档的元信息
_uploaded_docs: dict = {}
# RAG Pipeline 实例（按 年级+学科 缓存）
_rag_pipelines: dict = {}
# 记忆存储实例（延迟初始化）
_memory_short: Optional[object] = None
_memory_long: Optional[object] = None


def get_short_memory():
    """获取或创建短期记忆实例"""
    global _memory_short
    if _memory_short is None:
        from app.memory_store import ShortTermMemory
        _memory_short = ShortTermMemory()
    return _memory_short


def get_long_memory():
    """获取或创建长期记忆实例"""
    global _memory_long
    if _memory_long is None:
        from app.memory_store import LongTermMemory
        _memory_long = LongTermMemory()
    return _memory_long


def get_rag_pipeline(grade: str, subject: str):
    """获取或创建 RAG Pipeline——按年级+学科缓存，切换自动重建"""
    key = "{}_{}".format(grade, subject)
    if key not in _rag_pipelines:
        from app.edu_rag_engine import create_edu_rag
        _rag_pipelines[key] = create_edu_rag(
            connection_string=settings.database_url,
            grade=grade,
            subject=subject,
        )
    return _rag_pipelines[key]


# ================================================================
# 数据模型
# ================================================================

class EduChatRequest(BaseModel):
    """教育问答请求"""
    question: str = Field(..., min_length=1, description="学生问题")
    grade: str = Field(default="初三", description="年级")
    subject: str = Field(default="数学", description="学科")
    chapter: Optional[str] = Field(default=None, description="限定章节")
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: Optional[str] = Field(default=None, description="会话 ID（传入后支持历史上下文）")


class EduChatResponse(BaseModel):
    """教育问答响应"""
    answer: str
    grade: str
    subject: str
    sources: list = []


class ExamGenRequest(BaseModel):
    """试题生成请求"""
    context: str = Field(..., min_length=10, description="教材内容")
    question_type: str = Field(default="单选题", description="题型")
    count: int = Field(default=5, ge=1, le=20)
    difficulty: str = Field(default="中等")
    grade: str = Field(default="初三")
    subject: str = Field(default="数学")


class AgentChatRequest(BaseModel):
    """多 Agent 统一请求"""
    question: str = Field(..., min_length=1, description="用户问题")
    grade: str = Field(default="初三", description="年级")
    subject: str = Field(default="数学", description="学科")
    session_id: Optional[str] = Field(default=None, description="会话 ID")


class WrongAnswer(BaseModel):
    """错题记录"""
    knowledge_point: str = Field(default="未知")
    difficulty: str = Field(default="基础")


# ================================================================
# API 接口
# ================================================================

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "service": "智能教材问答与学习辅助平台",
    }


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Query(default="textbook", description="文档类型"),
    grade: str = Query(default="", description="年级（留空自动检测）"),
    subject: str = Query(default="", description="学科（留空自动检测）"),
):
    """上传教材/课件/题库

    上传的文件保存到 data 目录，支持 PDF/DOCX/MD/TXT
    """
    # 验证文件类型
    allowed_extensions = {".pdf", ".docx", ".md", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="不支持的文件格式: {}，支持: {}".format(
                ext, ", ".join(allowed_extensions)
            ),
        )

    # 安全文件名（防路径遍历）
    safe_name = _safe_filename(file.filename)

    # 确定保存目录
    type_dirs = {
        "textbook": "textbooks",
        "courseware": "courseware",
        "exam_bank": "exam_banks",
        "syllabus": "textbooks",
        "lesson_plan": "textbooks",
    }
    subdir = type_dirs.get(doc_type, "textbooks")
    save_dir = Path(settings.DATA_DIR) / subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存文件
    file_path = save_dir / safe_name
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 记录元信息
    _uploaded_docs[safe_name] = {
        "path": str(file_path),
        "doc_type": doc_type,
        "original_name": file.filename,
        "grade": grade if grade else None,
        "subject": subject if subject else None,
        "size": len(content),
    }

    return {
        "status": "ok",
        "filename": safe_name,
        "original_name": file.filename,
        "doc_type": doc_type,
        "path": str(file_path),
        "size_bytes": len(content),
    }


@app.post("/api/chat", response_model=EduChatResponse)
async def edu_chat(request: EduChatRequest):
    """教育问答接口 —— 年级/学科/章节可过滤，支持会话历史

    示例请求:
    ```json
    {
        "question": "一元二次方程的求根公式是什么？",
        "grade": "初三",
        "subject": "数学",
        "session_id": "abc123"
    }
    ```
    """
    try:
        pipeline = get_rag_pipeline(request.grade, request.subject)

        # 如果有 session_id，加载历史上下文
        history_context = ""
        if request.session_id:
            memory = get_short_memory()
            history_context = await memory.build_prompt_context(
                request.session_id
            )

        # 执行查询（注入历史上下文）
        result = pipeline.query(request.question, history_context=history_context)

        # 保存本轮对话到短期记忆
        if request.session_id:
            memory = get_short_memory()
            await memory.save_context(
                session_id=request.session_id,
                user_msg=request.question,
                agent_msg=result["answer"],
                sources=result.get("sources", []),
            )

        return EduChatResponse(
            answer=result["answer"],
            grade=result["grade"],
            subject=result["subject"],
            sources=result["sources"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="问答服务异常: {}".format(str(e)),
        )


@app.get("/api/chat/history")
async def get_chat_history(session_id: str = Query(..., description="会话 ID")):
    """获取会话历史 —— 刷新页面后恢复聊天记录

    返回最近 50 轮对话，前端用于恢复聊天界面。
    """
    try:
        memory = get_short_memory()
        raw = await memory.redis.lrange(
            "chat:{}".format(session_id), -50, -1
        )
        history = []
        for r in raw:
            entry = json.loads(r)
            history.append({
                "role": "user",
                "content": entry.get("user", ""),
            })
            history.append({
                "role": "assistant",
                "content": entry.get("agent", ""),
                "sources": entry.get("sources", []),
            })
        return {"session_id": session_id, "messages": history}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="获取历史失败: {}".format(str(e)),
        )


@app.post("/api/agent")
async def agent_chat(request: AgentChatRequest):
    """统一 Agent 接口 —— 自动识别意图 + 多步编排 + 记忆支持

    示例请求:
    ```json
    {
        "question": "帮我查勾股定理在哪一章，再出3道题",
        "grade": "初三",
        "subject": "数学",
        "session_id": "abc123"
    }
    ```
    """
    import asyncio
    from app.orchestrator_agent import OrchestratorAgent

    try:
        # 如果有 session_id，加载历史上下文
        enriched_question = request.question
        if request.session_id:
            memory = get_short_memory()
            history = await memory.build_prompt_context(request.session_id)
            if history:
                enriched_question = "{}\n\n当前新问题：{}".format(
                    history.strip(), request.question
                )

        # 同步方法在线程池运行，不阻塞事件循环
        loop = asyncio.get_running_loop()
        agent = OrchestratorAgent()
        result = await loop.run_in_executor(
            None, agent.run,
            enriched_question, request.grade,
            request.subject, request.session_id or "",
        )

        # 保存本轮对话到短期记忆
        if request.session_id:
            memory = get_short_memory()
            await memory.save_context(
                session_id=request.session_id,
                user_msg=request.question,
                agent_msg=result["answer"],
            )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Agent 服务异常: {}".format(str(e)),
        )


@app.post("/api/exam/generate")
async def generate_exam(request: ExamGenRequest):
    """试题生成接口

    示例请求:
    ```json
    {
        "context": "一元二次方程 ax²+bx+c=0 的求根公式...",
        "question_type": "单选题",
        "count": 5,
        "difficulty": "中等",
        "grade": "初三",
        "subject": "数学"
    }
    ```
    """
    from app.edu_features import ExamGenerator
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=0.3,  # 试题生成需要一定的创造性
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_API_BASE,
    )

    generator = ExamGenerator(llm)
    questions = generator.generate(
        context=request.context,
        question_type=request.question_type,
        count=request.count,
        difficulty=request.difficulty,
        grade=request.grade,
        subject=request.subject,
    )

    return {
        "questions": questions,
        "count": len(questions),
        "question_type": request.question_type,
        "difficulty": request.difficulty,
    }


@app.post("/api/image/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    grade: str = Query(default="", description="年级（留空自动检测）"),
    subject: str = Query(default="", description="学科（留空自动检测）"),
):
    """上传图片 → 自动识别内容

    支持 JPG/PNG，使用百炼全模态模型提取图片中的文字和公式
    """
    from app.multimodal_processor import ImageProcessor

    # 验证图片格式
    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail="不支持的图片格式: {}，支持: {}".format(
                ext, ", ".join(allowed)
            ),
        )

    # 保存临时文件
    safe_name = _safe_filename(file.filename)
    img_dir = Path(settings.DATA_DIR) / "temp"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = str(img_dir / safe_name)
    with open(img_path, "wb") as f:
        f.write(await file.read())

    try:
        processor = ImageProcessor()
        result = processor.extract_text(img_path)
        return {"text": result, "filename": file.filename}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="图片分析失败: {}".format(str(e)),
        )
    finally:
        # 清理临时文件
        if os.path.exists(img_path):
            os.remove(img_path)


@app.post("/api/image/solve")
async def solve_math_photo(
    file: UploadFile = File(...),
    grade: str = Query(default="初三"),
    subject: str = Query(default="数学"),
):
    """拍照解题：上传题目图片 → 自动识别 → Agent 解答

    支持 JPG/PNG，自动识别题目并调用 Agent 解答
    """
    from app.multimodal_processor import ImageProcessor

    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail="不支持的图片格式: {}，支持: {}".format(
                ext, ", ".join(allowed)
            ),
        )

    safe_name = _safe_filename(file.filename)
    img_dir = Path(settings.DATA_DIR) / "temp"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = str(img_dir / safe_name)
    with open(img_path, "wb") as f:
        f.write(await file.read())

    try:
        processor = ImageProcessor()
        result = processor.solve_math_problem(
            img_path, grade=grade, subject=subject
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="拍照解题失败: {}".format(str(e)),
        )
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


@app.get("/api/knowledge/graph")
async def knowledge_graph(
    grade: str = Query(default="初三"),
    subject: str = Query(default="数学"),
):
    """获取知识点图谱（示例数据，实际应持久化存储）"""
    return {
        "nodes": [
            {"id": "一元一次方程", "group": "预备知识"},
            {"id": "一元二次方程", "group": "核心知识"},
            {"id": "判别式", "group": "核心知识"},
            {"id": "韦达定理", "group": "拓展知识"},
            {"id": "二次函数", "group": "关联知识"},
        ],
        "edges": [
            {"from": "一元一次方程", "to": "一元二次方程", "relation": "前置依赖"},
            {"from": "一元二次方程", "to": "判别式", "relation": "包含"},
            {"from": "一元二次方程", "to": "韦达定理", "relation": "包含"},
            {"from": "一元二次方程", "to": "二次函数", "relation": "关联"},
        ],
        "grade": grade,
        "subject": subject,
    }


@app.post("/api/study/analyze")
async def analyze_wrong_answers(
    wrong_questions: List[WrongAnswer],
    grade: str = Query(default="初三"),
    subject: str = Query(default="数学"),
):
    """错题分析 —— 用 RAG 在教材中匹配真实知识点

    输入错题内容，系统自动在教材中检索对应知识点并给出复习建议。
    """
    from app.edu_features import StudyAnalyzer

    questions = [q.model_dump() for q in wrong_questions]

    # 对每个错题，用 RAG 在教材中检索真实知识点
    enriched = []
    for q in questions:
        raw = q.get("knowledge_point", "")
        # 用原始输入作为查询，在向量库中找相关知识点
        real_kp = _find_knowledge_point(raw, grade, subject)
        enriched.append({
            "original": raw,
            "knowledge_point": real_kp or raw,
            "difficulty": q.get("difficulty", "基础"),
        })

    result = StudyAnalyzer.analyze_wrong_answers(enriched)
    result["matched"] = [
        {"input": e["original"], "matched_kp": e["knowledge_point"]}
        for e in enriched
    ]
    return result


def _find_knowledge_point(query: str, grade: str, subject: str) -> str:
    """用 LLM 推理错题对应的真实知识点，并查找所属章节"""
    try:
        from langchain_openai import ChatOpenAI
        from app.config import settings
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_API_BASE,
        )
        prompt = (
            "你是{0}的{1}老师。学生做错了这道题或知识点，"
            "请用5个字以内说出对应的教材知识点名称。"
            "只输出知识点名，不要解释。\n错题：{2}"
        ).format(grade, subject, query)
        resp = llm.invoke(prompt)
        kp = resp.content.strip().replace("知识点：", "").replace("：", "")
        if len(kp) < 2 or len(kp) > 30:
            return query

        # 在教材中查找该知识点所属章节
        chapter = _find_chapter_for_kp(kp, grade, subject)
        if chapter:
            return "{0}（{1}）".format(kp, chapter)
        return kp
    except Exception:
        pass
    return query


def _find_chapter_for_kp(kp: str, grade: str, subject: str) -> str:
    """在教材中查找知识点所属章节"""
    try:
        from app.edu_rag_engine import EduVectorStore
        from app.config import settings
        store = EduVectorStore(settings.database_url)
        retriever = store.get_retriever(grade=grade, subject=subject, k=1, score_threshold=0.25)
        docs = retriever.invoke(kp)
        if docs:
            ch = docs[0].metadata.get("chapter")
            if ch:
                return ch
    except Exception:
        pass
    return ""


@app.get("/api/documents")
async def list_documents():
    """列出已上传的文档"""
    docs = []
    data_dir = Path(settings.DATA_DIR)
    for subdir in ["textbooks", "courseware", "exam_banks"]:
        d = data_dir / subdir
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    docs.append({
                        "filename": f.name,
                        "type": subdir,
                        "size_bytes": f.stat().st_size,
                        "path": str(f),
                    })
    return {"documents": docs, "total": len(docs)}


@app.post("/api/documents/delete")
async def delete_document(filename: str):
    """删除指定文档"""
    safe_name = _safe_filename(filename)
    data_dir = Path(settings.DATA_DIR)
    for subdir in ["textbooks", "courseware", "exam_banks"]:
        file_path = data_dir / subdir / safe_name
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            return {"status": "ok", "message": "已删除: {}".format(filename)}
    raise HTTPException(status_code=404, detail="文件不存在: {}".format(filename))


class IngestRequest(BaseModel):
    """文档入库请求"""
    doc_type: str = Field(default="textbook", description="文档类型")
    grade: str = Field(default="初三", description="年级")
    subject: str = Field(default="数学", description="学科")


@app.post("/api/ingest")
async def ingest_documents(request: IngestRequest):
    """一键导入文档到向量库

    将 data 目录下已上传的文档加载、切分、向量化后存入 pgvector。
    """
    from app.edu_document_loader import load_from_directory
    from app.edu_splitter import split_documents
    from app.edu_rag_engine import EduVectorStore

    type_dirs = {
        "textbook": "textbooks",
        "courseware": "courseware",
        "exam_bank": "exam_banks",
    }
    subdir = type_dirs.get(request.doc_type, "textbooks")
    data_path = Path(settings.DATA_DIR) / subdir

    if not data_path.exists():
        raise HTTPException(status_code=404, detail="目录不存在: {}".format(subdir))

    try:
        # 加载
        docs = load_from_directory(
            dir_path=str(data_path),
            doc_type=request.doc_type,
            grade=request.grade,
            subject=request.subject,
        )
        if not docs:
            raise HTTPException(status_code=404, detail="未找到文档文件，请先上传")

        # 切分
        chunks = split_documents(docs)

        # 向量化存入
        store = EduVectorStore(settings.database_url)
        store.create_collection(
            documents=chunks,
            grade=request.grade,
            subject=request.subject,
        )

        return {
            "status": "ok",
            "grade": request.grade,
            "subject": request.subject,
            "doc_type": request.doc_type,
            "raw_pages": len(docs),
            "chunks": len(chunks),
            "message": "成功导入 {0} 个知识块".format(len(chunks)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="导入失败: {}".format(str(e)))


# ================================================================
# 启动入口
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  [SmartTextbook] 智能教材问答与学习辅助平台")
    print("  FastAPI + pgvector + LangChain + DeepSeek")
    print("=" * 60)
    print("  API 文档: http://{}:{}/docs".format(
        settings.APP_HOST, settings.APP_PORT
    ))
    print("  LLM: {} ({})".format(settings.LLM_MODEL, settings.LLM_API_BASE))
    print("  Embedding: {}".format(settings.EMBEDDING_MODEL))
    print("=" * 60)

    uvicorn.run(
        "app.edu_api:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=True,
    )
