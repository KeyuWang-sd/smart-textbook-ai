# edu_features.py
"""教育领域特有功能：试题生成、知识点图谱、错题分析"""

import json
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ================================================================
# 1. 智能试题生成器
# ================================================================

class ExamGenerator:
    """智能试题生成器 —— 根据教材内容出题"""

    QUESTION_TYPES = ["单选题", "多选题", "填空题", "判断题", "简答题"]

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def generate(
        self,
        context: str,
        question_type: str = "单选题",
        count: int = 3,
        difficulty: str = "中等",
        grade: str = "初三",
        subject: str = "数学",
    ) -> List[Dict]:
        """根据教材内容生成试题

        Args:
            context: 教材内容上下文
            question_type: 题型
            count: 出题数量
            difficulty: 难度（基础/中等/困难）
            grade: 年级
            subject: 学科

        Returns:
            List[Dict]: 试题列表
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是{grade}的{subject}老师。请根据提供的教材内容出题。

出题要求：
1. 题型：{question_type}
2. 数量：{count} 道
3. 难度：{difficulty}
4. 每道题必须基于上下文中的具体知识点
5. 严格输出 JSON 数组格式，不要包含任何其他文字。每题包含以下字段：
   - "question": "题目",
   - "options": ["选项A", "选项B", "选项C", "选项D"] (非选择题填空为[]),
   - "answer": "正确答案",
   - "explanation": "解析",
   - "knowledge_point": "考察知识点"

教材内容：
{context}

请直接输出 JSON 数组，不要包裹在 ```json``` 中："""),
            ("human", "请出题"),
        ])

        response = self.llm.invoke(
            prompt.format_messages(
                grade=grade,
                subject=subject,
                question_type=question_type,
                count=count,
                difficulty=difficulty,
                context=context[:3000],
            )
        )

        # 解析 JSON
        try:
            questions = json.loads(response.content)
            return questions
        except json.JSONDecodeError:
            # 尝试提取 JSON 部分
            import re
            match = re.search(r"\[.*\]", response.content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return [{"error": "解析失败", "raw": response.content}]


# ================================================================
# 2. 知识点关系图谱
# ================================================================

class KnowledgeGraph:
    """知识点关系图谱 —— 记录前置-后置依赖"""

    def __init__(self):
        # 知识点 -> 前置知识列表
        self.prerequisites: Dict[str, List[str]] = {}
        # 知识点 -> 后置知识列表
        self.follow_ups: Dict[str, List[str]] = {}

    def add_relation(self, pre: str, post: str):
        """添加前置-后置关系

        Args:
            pre: 前置知识点
            post: 后置知识点
        """
        self.prerequisites.setdefault(post, []).append(pre)
        self.follow_ups.setdefault(pre, []).append(post)

    def get_missing_prerequisites(
        self, topic: str, mastered: List[str]
    ) -> List[str]:
        """计算某个知识点还缺哪些前置知识"""
        prereqs = self.prerequisites.get(topic, [])
        return [p for p in prereqs if p not in mastered]

    def suggest_learning_path(
        self, target: str, mastered: List[str]
    ) -> List[str]:
        """推荐学习路径（BFS）

        Args:
            target: 目标知识点
            mastered: 已掌握的知识点列表

        Returns:
            List[str]: 建议学习路径
        """
        path = []
        to_learn = [target]
        while to_learn:
            current = to_learn.pop(0)
            if current in mastered or current in path:
                continue
            prereqs = self.get_missing_prerequisites(current, mastered)
            for p in prereqs:
                if p not in path:
                    to_learn.append(p)
            if current not in path:
                path.append(current)
        return path


# ================================================================
# 3. 学习分析器
# ================================================================

class StudyAnalyzer:
    """学习分析器 —— 错题分析 + 薄弱点诊断"""

    @staticmethod
    def analyze_wrong_answers(
        wrong_questions: List[Dict],
    ) -> Dict:
        """分析错题，输出薄弱知识点

        Args:
            wrong_questions: 错题列表，每题需含 knowledge_point 和 difficulty

        Returns:
            Dict: {
                "weak_points": list,        # 薄弱知识点排名
                "difficulty_distribution": dict, # 难度分布
                "suggestion": str,          # 复习建议
            }
        """
        knowledge_count: Dict[str, int] = {}
        difficulty_count: Dict[str, int] = {}

        for q in wrong_questions:
            kp = q.get("knowledge_point", "未知")
            knowledge_count[kp] = knowledge_count.get(kp, 0) + 1
            diff = q.get("difficulty", "未知")
            difficulty_count[diff] = difficulty_count.get(diff, 0) + 1

        # 高频错误知识点 → 薄弱点
        weak_points = sorted(
            knowledge_count.items(), key=lambda x: x[1], reverse=True
        )

        return {
            "weak_points": [
                {"knowledge": kp, "error_count": count}
                for kp, count in weak_points[:5]
            ],
            "difficulty_distribution": difficulty_count,
            "suggestion": "建议重点复习以下知识点：" + "、".join(
                [w[0] for w in weak_points[:3]]
            ),
        }
