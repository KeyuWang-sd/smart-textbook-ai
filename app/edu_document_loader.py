# edu_document_loader.py
"""教育领域文档加载器 —— 专门处理教材、课件、题库等教育文档

支持格式：PDF、DOCX、Markdown、TXT
自动识别：学科、年级
元数据注入：来源文件、页码、文档类型、年级、学科
"""

from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
import re


class EduDocumentLoader:
    """教育文档加载器 —— 为教育场景优化的元数据标注"""

    # 教育文档类型映射
    DOC_TYPES = {
        "textbook":    {"desc": "教材",     "icon": "📖"},
        "courseware":   {"desc": "课件",     "icon": "📊"},
        "exam_bank":    {"desc": "题库",     "icon": "📝"},
        "syllabus":     {"desc": "课程标准", "icon": "📋"},
        "lesson_plan":  {"desc": "教案",     "icon": "📑"},
    }

    # 学科识别关键词
    SUBJECT_KEYWORDS = {
        "语文": ["语文", "阅读", "作文", "文言文", "古诗词"],
        "数学": ["数学", "函数", "方程", "几何", "代数", "概率"],
        "英语": ["英语", "English", "语法", "词汇", "完形填空"],
        "物理": ["物理", "力学", "电学", "光学", "牛顿", "电场"],
        "化学": ["化学", "反应", "元素", "分子", "摩尔"],
        "生物": ["生物", "细胞", "遗传", "生态", "光合作用"],
        "历史": ["历史", "朝代", "战争", "改革", "革命"],
        "地理": ["地理", "气候", "地形", "人口", "区域"],
        "政治": ["政治", "宪法", "制度", "权利", "义务"],
    }

    @classmethod
    def detect_subject(cls, text: str) -> str:
        """从文本内容自动识别学科"""
        scores = {}
        for subject, keywords in cls.SUBJECT_KEYWORDS.items():
            scores[subject] = sum(text.count(kw) for kw in keywords)
        if max(scores.values()) == 0:
            return "通用"
        return max(scores, key=scores.get)

    @classmethod
    def detect_grade(cls, text: str) -> str:
        """从文本检测适用年级"""
        grade_patterns = [
            (r"小学([一二三四五六])年级", lambda m: "小学{}年级".format(m.group(1))),
            (r"初([一二三])", lambda m: "初{}".format(m.group(1))),
            (r"高([一二三])", lambda m: "高{}".format(m.group(1))),
            (r"(七年级|八年级|九年级)", lambda m: m.group(1)),
            (r"(高一|高二|高三)", lambda m: m.group(1)),
        ]
        for pattern, extract in grade_patterns:
            match = re.search(pattern, text)
            if match:
                return extract(match)
        return "未标注"

    @classmethod
    def load(
        cls,
        file_path: str,
        doc_type: str = "textbook",
        grade: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> List[Document]:
        """加载教育文档，注入教育元数据

        Args:
            file_path: 文档路径
            doc_type: 文档类型 (textbook/courseware/exam_bank/syllabus/lesson_plan)
            grade: 年级（可选，不传则自动检测）
            subject: 学科（可选，不传则自动检测）

        Returns:
            List[Document]: 带元数据的文档列表
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        # 根据格式选择加载器
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix == ".docx":
            loader = UnstructuredWordDocumentLoader(str(path))
        elif suffix in (".md", ".txt"):
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            raise ValueError("不支持格式: {}".format(suffix))

        docs = loader.load()
        full_text = " ".join([d.page_content[:500] for d in docs])

        # 自动检测学科和年级
        detected_subject = subject or cls.detect_subject(full_text)
        detected_grade = grade or cls.detect_grade(full_text)

        for i, doc in enumerate(docs):
            doc.metadata.update({
                "source_file": path.name,
                "file_type": suffix,
                "doc_type": doc_type,
                "doc_type_label": cls.DOC_TYPES.get(doc_type, {}).get("desc", "未知"),
                "grade": detected_grade,
                "subject": detected_subject,
                "page_index": i,  # 教育场景常需要页码
            })

        return docs


# ============================================================
# 便捷函数：从目录批量加载
# ============================================================

def load_from_directory(
    dir_path: str,
    doc_type: str = "textbook",
    grade: Optional[str] = None,
    subject: Optional[str] = None,
    extensions: List[str] = None,
) -> List[Document]:
    """从目录批量加载教育文档"""
    if extensions is None:
        extensions = [".pdf", ".docx", ".md", ".txt"]

    all_docs = []
    dir_p = Path(dir_path)

    for ext in extensions:
        for file_path in dir_p.glob("*{}".format(ext)):
            try:
                docs = EduDocumentLoader.load(
                    str(file_path),
                    doc_type=doc_type,
                    grade=grade,
                    subject=subject,
                )
                all_docs.extend(docs)
                print("[OK] 已加载: {} ({} 页)".format(
                    file_path.name, len(docs)
                ))
            except Exception as e:
                print("[SKIP] 跳过 {}: {}".format(file_path.name, e))

    print("\n总计加载 {} 个文档块".format(len(all_docs)))
    return all_docs
