# edu_rag_engine.py
"""教育领域 RAG 引擎 —— 支持年级/学科/章节/知识点四维过滤 + 难度自适应

核心流程:
① 查询扩展 (Multi-Query) → 适配学生用语
② 向量检索 (pgvector)   → 元数据过滤
③ 去重                   → 合并多查询结果
④ 重排序 (API Reranker)  → 年级匹配加权 +15%
⑤ 上下文构建             → 章节引用
⑥ LLM 生成               → 教育化 Prompt
"""

import os
from typing import List, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# ================================================================
# 可选的 requests 库；若未安装则 fallback
# ================================================================
try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ================================================================
# Embedding 配置 —— 使用 text-embedding-v3 + 百炼 OpenAI 兼容接口
# ================================================================

DEFAULT_EMBEDDING_MODEL = "text-embedding-v3"
# 百炼 OpenAI 兼容模式 API
BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# v3 支持可变维度: 1024(默认) / 768 / 512
DEFAULT_EMBEDDING_DIMENSIONS = 1024


def _get_embeddings():
    """创建 Embeddings 实例 —— 使用百炼 OpenAI 兼容接口"""
    return OpenAIEmbeddings(
        model=DEFAULT_EMBEDDING_MODEL,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=BAILIAN_BASE_URL,
        tiktoken_enabled=False,          # 百炼 API 需要原始文本，不能传 token ID
        check_embedding_ctx_length=False,  # 跳过上下文长度检查
    )


# ================================================================
# 1. 教育化向量存储
# ================================================================

class EduVectorStore:
    """教育场景向量存储 —— 元数据过滤 + 年级/学科索引"""

    def __init__(self, connection_string: str):
        """
        Args:
            connection_string: PostgreSQL 连接串
                e.g. postgresql://postgres:***@localhost:5432/edu_knowledge
        """
        self.connection = connection_string
        self.embeddings = _get_embeddings()

    def create_collection(
        self,
        documents: List[Document],
        grade: str,
        subject: str,
    ) -> PGVector:
        """创建年级-学科维度的知识库

        每个 (年级, 学科) 组合一个 pgvector collection，
        方便按年级+学科快速过滤检索。
        """
        collection = "edu_{}_{}".format(grade, subject)
        # pgvector collection 名不能有特殊字符
        collection = collection.replace(" ", "_").replace("/", "_")

        # 分批导入，百炼限制每次最多10条
        batch_size = 10
        last_store = None
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            last_store = PGVector.from_documents(
                documents=batch,
                embedding=self.embeddings,
                connection=self.connection,
                collection_name=collection,
            )
        return last_store

    def get_retriever(
        self,
        grade: str,
        subject: str,
        k: int = 5,
        chapter: Optional[str] = None,
        score_threshold: float = 0.5,
    ):
        """获取教育场景检索器 —— 支持元数据过滤

        Args:
            grade: 年级
            subject: 学科
            k: 返回文档数
            chapter: 可选，限定章节
        """
        collection = "edu_{}_{}".format(grade, subject).replace(" ", "_")

        vector_store = PGVector(
            connection=self.connection,
            embeddings=self.embeddings,
            collection_name=collection,
        )

        search_kwargs = {"k": k, "score_threshold": score_threshold}

        # 章节过滤：只检索指定章节的内容
        if chapter:
            search_kwargs["filter"] = {"chapter": chapter}

        return vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs,
        )


# ================================================================
# 2. 教育场景重排序 (Rerank)
# ================================================================

class DashScopeReranker:
    """百炼 DashScope API 重排序 —— 用 gte-rerank-v2 在线模型

    通过百炼 OpenAI 兼容接口调用 rerank API，无需本地下载模型。
    支持年级匹配加权（同年级文档 +15% 权重）。
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gte-rerank-v2",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ):
        """
        Args:
            api_key: 百炼 DashScope API Key（settings.DASHSCOPE_API_KEY）
            model: rerank 模型名，默认 gte-rerank-v2
            base_url: OpenAI 兼容接口地址
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._available = bool(api_key and HAS_REQUESTS)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3,
        target_grade: Optional[str] = None,
    ) -> List[Document]:
        """调用百炼 rerank API 重排序 + 年级匹配加权

        Args:
            query: 学生提问
            documents: 候选文档列表
            top_k: 返回 top-k 文档
            target_grade: 目标年级，同年级文档 +15% 权重

        Returns:
            List[Document]: 重排序后的文档列表
        """
        if not self._available or not documents:
            return documents[:top_k]

        # 准备请求
        url = "{}/rerank".format(self.base_url)
        headers = {
            "Authorization": "Bearer {}".format(self.api_key),
            "Content-Type": "application/json",
        }
        # top_n 多取一些，留加权排序的余量
        top_n = min(top_k * 2 if top_k > 1 else top_k + 1, len(documents))
        payload = {
            "model": self.model,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": top_n,
        }

        try:
            resp = _requests.post(
                url, headers=headers, json=payload, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

            # 兼容两种返回格式（OpenAI 兼容模式 / DashScope 原生格式）
            results = data.get("results") or data.get("output", {}).get("results", [])
            if not results:
                return documents[:top_k]

            # 解析分数并按年级加权
            scored_docs = []
            for r in results:
                idx = r.get("index", 0)
                if idx >= len(documents):
                    continue
                score = r.get("relevance_score", r.get("score", 0.0))
                doc = documents[idx]

                # 教育场景：同年级文档 +15% 权重
                doc_grade = doc.metadata.get("grade", "")
                if target_grade and doc_grade == target_grade:
                    score *= 1.15

                scored_docs.append((doc, score))

            if not scored_docs:
                return documents[:top_k]

            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[:top_k]]

        except Exception as e:
            print("[DashScopeReranker] API 调用失败: {}，回退取 top-{}".format(
                e, top_k
            ))
            return documents[:top_k]


# 兼容旧名，方便已有引用的代码
EduReranker = DashScopeReranker


# ================================================================
# 3. 教育场景 Multi-Query (多角度查询扩展)
# ================================================================

class EduMultiQuery:
    """教育场景查询扩展 —— 生成适合学生理解的变体"""

    def __init__(self, llm):
        self.llm = llm

    def expand(
        self, question: str, grade: str, n: int = 3
    ) -> List[str]:
        """生成多个查询变体，适配学生用语习惯"""
        prompt = """你是一个{grade}学生的助教。请将以下问题改写为 {n} 个不同角度的版本：

改写规则：
1. 保持原意，使用{grade}学生能理解的词汇和表述
2. 尝试不同的提问方式（直接问、追问、场景化提问）
3. 如果原问题是教材中的术语，生成1个更口语化的版本

原始问题：{question}

请每行输出一个改写版本，不要编号：""".format(
            grade=grade, n=n, question=question
        )

        response = self.llm.invoke(prompt)
        variations = [
            q.strip()
            for q in response.content.split("\n")
            if q.strip()
        ]
        return [question] + variations[:n]


# ================================================================
# 4. 教育场景系统 Prompt
# ================================================================

EDUCATION_SYSTEM_PROMPT = """你是一个专业的{subject}学科教育智能助教，服务对象是{grade}学生。

=== 回答规则 ===
1. 只使用教材上下文，不编造知识
2. 用{grade}学生能理解的语言，避免超纲术语
3. 复杂概念分步骤讲解，先基础再深入
4. 末尾注明知识点出处（教材第几章）
5. 回答后追问一个引导性问题，帮助加深理解
6. 公式必须用纯文本，禁止使用 LaTeX（如 \\frac、\\sqrt、\\[ 等）
   正确示例：x = (-b ± √(b²-4ac)) / 2a
   错误示例：\\[ x = \\frac{{-b \\pm \\sqrt{{b^2 - 4ac}}}}{{2a}} \\]

=== 回答格式 ===
知识型问题：直接给出答案 → 通俗解释原理 → 举一个简单例子 → 注明教材出处
试题类问题：先分析解题思路 → 展示步骤和答案 → 总结涉及的知识点

=== 教材参考 ===
{context}
"""


# ================================================================
# 5. 教育化 RAG Pipeline
# ================================================================

class EduRAGPipeline:
    """教育 RAG 完整流程"""

    def __init__(
        self,
        llm,
        retriever,
        reranker: EduReranker,
        expander: EduMultiQuery,
        grade: str,
        subject: str,
    ):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.expander = expander
        self.grade = grade
        self.subject = subject

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", EDUCATION_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

    def query(self, question: str, history_context: str = "") -> dict:
        """执行教育化 RAG 查询

        Args:
            question: 学生提问
            history_context: 可选，历史对话上下文（来自短期记忆）

        Returns:
            dict: {
                "answer": str,           # LLM 生成的回答
                "grade": str,            # 年级
                "subject": str,          # 学科
                "sources": list[dict],   # 引用来源
                "variations": list[str], # 查询变体
            }
        """
        # ① 查询扩展（适配学生用语）
        variations = self.expander.expand(question, self.grade)

        # ② 对所有变体检索
        all_docs = []
        for vq in variations:
            docs = self.retriever.invoke(vq)
            all_docs.extend(docs)

        # ③ 去重
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        # ④ 教育化重排序（考虑年级匹配）- 如 reranker 不可用则跳过
        try:
            top_docs = self.reranker.rerank(
                question,
                unique_docs,
                top_k=3,
                target_grade=self.grade,
            )
        except Exception:
            top_docs = unique_docs[:3]

        # ⑤ 构建带章节引用的上下文（无相关文档时直接回答）
        if not top_docs:
            return {
                "answer": "抱歉，教材中没有找到与「{}」相关的内容。请尝试换个问法，或上传更多教材资料。".format(question),
                "grade": self.grade,
                "subject": self.subject,
                "sources": [],
                "variations": variations,
            }

        context_parts = []
        for d in top_docs:
            chapter = d.metadata.get("chapter", "")
            source = d.metadata.get("source_file", "教材")
            header = "[{}]".format(source)
            if chapter:
                header += " {}".format(chapter)
            context_parts.append("{}\n{}".format(header, d.page_content))

        context = "\n\n---\n\n".join(context_parts)

        # ⑥ 注入历史上下文（如有）
        enriched_question = question
        if history_context:
            enriched_question = "{}\n\n当前问题：{}".format(
                history_context.strip(), question
            )

        # ⑦ 生成回答
        response = self.llm.invoke(
            self.prompt.format_messages(
                subject=self.subject,
                grade=self.grade,
                context=context,
                question=enriched_question,
            )
        )

        return {
            "answer": _latex_to_text(response.content),
            "grade": self.grade,
            "subject": self.subject,
            "sources": [
                {
                    "content": d.page_content[:200],
                    "chapter": d.metadata.get("chapter", ""),
                    "source_file": d.metadata.get("source_file", ""),
                }
                for d in top_docs
            ],
            "variations": variations,
        }


def _latex_to_text(text: str) -> str:
    """将 LaTeX 公式转为纯文本"""
    import re
    # 去掉 inline 和 display math 包裹
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("\\(", "").replace("\\)", "")
    # LaTeX 命令转纯文本
    text = text.replace("\\frac{", "(")
    text = text.replace("}{", ")/(")
    text = text.replace("\\sqrt{", "\u221a(")
    text = text.replace("\\pm", "\u00b1")
    text = text.replace("\\neq", "\u2260")
    text = text.replace("\\times", "\u00d7")
    text = text.replace("\\div", "\u00f7")
    text = text.replace("\\cdot", "\u00b7")
    text = text.replace("\\leq", "\u2264")
    text = text.replace("\\geq", "\u2265")
    text = text.replace("\\quad", " ")
    text = text.replace("\\Delta", "\u0394")
    text = text.replace("\\pi", "\u03c0")
    text = text.replace("\\alpha", "\u03b1")
    text = text.replace("\\beta", "\u03b2")
    # 清理多余的 }
    text = text.replace("}", ")")
    text = re.sub(r"\\([a-zA-Z]+)", r"\1", text)
    return text.strip()


# ================================================================
# 工厂函数：创建 RAG Pipeline
# ================================================================

def create_edu_rag(
    connection_string: str,
    grade: str = "初三",
    subject: str = "数学",
    chapter: Optional[str] = None,
    k: int = 5,
    llm_model: str = "deepseek-v4-flash",
    llm_api_key: Optional[str] = None,
    llm_api_base: str = "https://api.deepseek.com/v1",
) -> EduRAGPipeline:
    """一站式创建教育 RAG Pipeline

    Args:
        connection_string: PostgreSQL 连接串
        grade: 目标年级
        subject: 目标学科
        chapter: 可选，限定章节
        k: 检索返回数
        llm_model: LLM 模型名
        llm_api_key: API Key
        llm_api_base: API Base URL

    Returns:
        EduRAGPipeline: 配置好的 Pipeline
    """
    # 创建 LLM
    llm = ChatOpenAI(
        model=llm_model,
        temperature=0.1,  # 教育场景需要更稳定的输出
        api_key=llm_api_key or os.getenv("LLM_API_KEY"),
        base_url=llm_api_base or os.getenv("LLM_API_BASE"),
    )

    # 创建向量存储和检索器
    vector_store = EduVectorStore(connection_string)
    retriever = vector_store.get_retriever(
        grade=grade,
        subject=subject,
        k=k,
        chapter=chapter,
    )

    # 创建重排序器（百炼 gte-rerank-v2 在线 API）
    reranker = DashScopeReranker(
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        model="gte-rerank-v2",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 创建查询扩展器
    expander = EduMultiQuery(llm)

    # 组装 Pipeline
    return EduRAGPipeline(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
        expander=expander,
        grade=grade,
        subject=subject,
    )
