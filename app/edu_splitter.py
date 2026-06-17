# edu_splitter.py
"""教育文档切分策略 —— 按教材层级结构切分

核心原则:
- 教材：保留章-节-知识点层级，chunk_size 偏大（800）因为教材逻辑连续
- 题库：按题号切分，每道题独立单元，保留题号/题型/难度/答案/解析
- 课件：按页/幻灯片切分，保留配图和要点
"""

from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


class EduDocumentSplitter:
    """教育文档专用切分器"""

    # 教材章节标题正则（兼容 markdown # 前缀）
    CHAPTER_PATTERN = re.compile(
        r"^(?:#{1,3}\s*)?(第[一二三四五六七八九十百零\d]+章|Chapter\s*\d+|Unit\s*\d+)\s"
    )
    SECTION_PATTERN = re.compile(
        r"^(?:#{1,4}\s*)?(第[一二三四五六七八九十百零\d]+节|\d+\.\d+|Section\s*\d+)\s"
    )
    # 知识点编号（如 1.2.3、知识点一）
    KNOWLEDGE_POINT = re.compile(
        r"(知识点[一二三四五六七八九十\d]+|\d+\.\d+\.\d+)"
    )

    # 题号匹配
    QUESTION_NUM = re.compile(r"^(第?\d+[．.、题])|(^\d+[．.)])")

    @classmethod
    def split_textbook(cls, docs: List[Document]) -> List[Document]:
        """教材切分：保留章节结构，大块切分保证知识完整性"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
            separators=[
                "\n\n\n", "\n## ", "\n### ",
                "\n\n", "\n", "。", "！", "？", "；", " ",
            ],
        )
        chunks = splitter.split_documents(docs)

        # 从每个chunk自身提取章节
        import re as _re
        sec_pattern = _re.compile(r'(?:###?\s*)?(\d+)\.(\d+)')
        for chunk in chunks:
            m = cls.CHAPTER_PATTERN.search(chunk.page_content[:200])
            if m:
                chapter = chunk.page_content[m.start():m.end()].strip()
                idx = chapter.index(m.group(1))
                chunk.metadata["chapter"] = chapter[idx:]
            else:
                # 尝试从节号推断（如 21.6 → 第21章）
                sm = sec_pattern.search(chunk.page_content[:100])
                if sm:
                    chunk.metadata["chapter"] = "第{}章".format(sm.group(1))
                else:
                    chunk.metadata["chapter"] = None

        return chunks

    @classmethod
    def split_exam_bank(cls, docs: List[Document]) -> List[Document]:
        """题库切分：按题目切分，保留题型/难度/答案/解析"""
        full_text = "\n".join([d.page_content for d in docs])

        # 按题号分割
        questions = cls._split_by_question(full_text)
        chunks = []

        for q in questions:
            doc = Document(
                page_content=q["content"],
                metadata={
                    "question_number": q["number"],
                    "question_type": cls._detect_question_type(q["content"]),
                    "difficulty": cls._detect_difficulty(q["content"]),
                    "has_answer": q.get("has_answer", False),
                    "knowledge_points": cls._extract_knowledge_points(
                        q["content"]
                    ),
                }
            )
            chunks.append(doc)

        return chunks

    @classmethod
    def split_courseware(cls, docs: List[Document]) -> List[Document]:
        """课件切分：按页切分，保持要点结构"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；", " "],
        )
        return splitter.split_documents(docs)

    # ---- 题库辅助方法 ----

    @classmethod
    def _split_by_question(cls, text: str) -> List[Dict]:
        """按题号分割题目"""
        lines = text.split("\n")
        questions = []
        current = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测是否新题开始
            if cls.QUESTION_NUM.match(line):
                if current:
                    questions.append(current)
                num_match = cls.QUESTION_NUM.search(line)
                number = num_match.group(0) if num_match else ""
                content = line[num_match.end():] if num_match else line
                current = {
                    "number": number,
                    "content": content + "\n",
                    "has_answer": False,
                }
            elif current:
                current["content"] += line + "\n"
                # 检测答案区域
                if re.search(r"(答案|解析|解答|参考答案)[：:]", line):
                    current["has_answer"] = True

        if current:
            questions.append(current)

        return questions

    @classmethod
    def _detect_question_type(cls, text: str) -> str:
        """识别题目类型"""
        type_keywords = {
            "单选题": ["A.", "B.", "C.", "D.", "正确选项"],
            "多选题": ["正确选项", "ABCD", "全选"],
            "判断题": ["正确", "错误", "对", "错", "\u221a", "\u00d7"],
            "填空题": ["__", "（  ）", "填空"],
            "简答题": ["简述", "说明", "请回答", "分析"],
            "计算题": ["计算", "求解", "证明", "推导"],
        }
        for qtype, keywords in type_keywords.items():
            if any(kw in text for kw in keywords):
                return qtype
        return "未知题型"

    @classmethod
    def _detect_difficulty(cls, text: str) -> str:
        """识别难度等级"""
        if re.search(r"(\u2605{3}|困难|拔高|拓展|竞赛)", text):
            return "困难"
        if re.search(r"(\u2605{2}|适中|中等|提高)", text):
            return "中等"
        return "基础"

    @classmethod
    def _extract_knowledge_points(cls, text: str) -> List[str]:
        """从题目提取关联知识点"""
        points = cls.KNOWLEDGE_POINT.findall(text)
        return [p for p in points]

    @classmethod
    def _extract_chapter_positions(cls, text: str):
        """提取章节位置列表 [(字符位置, 章节名), ...]"""
        positions = []
        for m in cls.CHAPTER_PATTERN.finditer(text):
            chapter_name = text[m.start():m.end()].strip()
            # 去掉 markdown # 前缀
            grp = m.group(1)
            idx = chapter_name.index(grp)
            chapter_name = chapter_name[idx:]
            positions.append((m.start(), chapter_name))
        return positions

    @classmethod
    def _extract_chapters(cls, text: str) -> Dict[str, str]:
        """提取教材章节结构"""
        chapters = {}
        for line in text.split("\n"):
            stripped = line.strip()
            m = cls.CHAPTER_PATTERN.match(stripped)
            if m:
                # 去掉 markdown # 前缀
                chapter_name = stripped[stripped.index(m.group(1)):]
                chapters[chapter_name[:30]] = chapter_name
        return chapters

    @classmethod
    def _find_chapter(
        cls, text: str, chapters: Dict[str, str]
    ) -> Optional[str]:
        """为文本块定位所属章节"""
        for key, chapter in chapters.items():
            if key[:10] in text or chapter[:10] in text:
                return chapter
        return None


# ============================================================
# 便捷函数：按文档类型自动分派切分策略
# ============================================================

def split_documents(docs: List[Document]) -> List[Document]:
    """根据元数据中的 doc_type 自动选择切分策略"""
    if not docs:
        return []

    doc_type = docs[0].metadata.get("doc_type", "textbook")

    if doc_type == "textbook":
        return EduDocumentSplitter.split_textbook(docs)
    elif doc_type == "exam_bank":
        return EduDocumentSplitter.split_exam_bank(docs)
    elif doc_type == "courseware":
        return EduDocumentSplitter.split_courseware(docs)
    else:
        # 默认用教材切分
        return EduDocumentSplitter.split_textbook(docs)
