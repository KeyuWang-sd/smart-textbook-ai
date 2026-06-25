# agent_tools.py
"""Agent Tool 定义层
每个 Tool 包装一个原子能力，供 Orchestrator Agent 调用
"""

from typing import Optional, Any
from pydantic import BaseModel


class ToolResult(BaseModel):
    """工具执行结果基类"""
    success: bool = True
    data: Any = None
    error: Optional[str] = None


class SearchTextbookTool:
    """工具1：查教材 —— 在教材中搜索知识点并回答"""
    name = "search_textbook"
    description = "在教材中搜索知识点并回答，适用于知识问答类问题"

    def __init__(self):
        self._pipelines: dict = {}

    def _get_pipeline(self, grade: str, subject: str):
        from app.config import settings
        from app.edu_rag_engine import create_edu_rag
        key = "{}_{}".format(grade, subject)
        if key not in self._pipelines:
            self._pipelines[key] = create_edu_rag(
                connection_string=settings.database_url,
                grade=grade,
                subject=subject,
            )
        return self._pipelines[key]

    def run(self, question: str, grade: str = "初三",
            subject: str = "数学", session_id: str = "") -> ToolResult:
        """执行教材检索"""
        try:
            pipeline = self._get_pipeline(grade, subject)
            result = pipeline.query(question)
            return ToolResult(data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class GenerateExamTool:
    """工具2：出题 —— 根据教材内容生成试题"""
    name = "generate_exam"
    description = "根据教材内容生成试题，适用于考试出题场景"

    def run(self, context: str, count: int = 5,
            question_type: str = "单选题",
            difficulty: str = "中等", grade: str = "初三",
            subject: str = "数学") -> ToolResult:
        """执行试题生成"""
        try:
            from langchain_openai import ChatOpenAI
            from app.config import settings
            from app.edu_features import ExamGenerator

            llm = ChatOpenAI(
                model=settings.LLM_MODEL,
                temperature=0.3,
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_API_BASE,
            )
            generator = ExamGenerator(llm)
            questions = generator.generate(
                context=context,
                question_type=question_type,
                count=count,
                difficulty=difficulty,
                grade=grade,
                subject=subject,
            )
            return ToolResult(data=questions)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class AnalyzeStudyTool:
    """工具3：错题分析 —— 分析错题，定位薄弱知识点"""
    name = "analyze_study"
    description = "分析错题，定位薄弱知识点，适用于复习建议场景"

    def run(self, wrong_answers: list, grade: str = "初三",
            subject: str = "数学") -> ToolResult:
        """执行错题分析"""
        try:
            from app.edu_features import StudyAnalyzer
            result = StudyAnalyzer.analyze_wrong_answers(wrong_answers)
            return ToolResult(data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class CalculatorTool:
    """工具4：数学计算 —— 使用 sympy 进行符号计算"""
    name = "calculator"
    description = "数学计算，适用于解方程、数值计算、公式推导"

    def run(self, expression: str) -> ToolResult:
        """执行数学计算"""
        try:
            import sympy as sp
            result = sp.simplify(expression)
            return ToolResult(data=str(result))
        except ImportError:
            return ToolResult(
                success=False,
                error="sympy 未安装，请执行: pip install sympy",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
