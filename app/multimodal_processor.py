# multimodal_processor.py
"""多模态输入处理器
支持：图片 → OCR → 文字提取 → 交给 Agent 处理
"""

import base64
from typing import Optional

from openai import OpenAI
from app.config import settings


class ImageProcessor:
    """图片处理器

    使用百炼的全模态模型（qwen3-omni-flash）识别图片中的文字和数学公式
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(
            api_key=api_key or settings.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def extract_text(self, image_path: str) -> str:
        """从图片中提取文字/题目

        Args:
            image_path: 图片文件路径

        Returns:
            str: 识别出的文本内容
        """
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode()

        response = self.client.chat.completions.create(
            model="qwen3-omni-flash",  # 百炼全模态模型
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请提取这张图片中的所有文字和数学公式，"
                                "保持原有格式和排版",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,{}".format(
                                base64_image
                            ),
                        },
                    },
                ],
            }],
            max_tokens=2048,
        )
        return response.choices[0].message.content

    def solve_math_problem(self, image_path: str,
                           grade: str = "初三",
                           subject: str = "数学") -> dict:
        """拍照解题：图片 → 识别题目 → Agent 解答

        Args:
            image_path: 题目图片路径
            grade: 年级
            subject: 学科

        Returns:
            dict: {
                "problem": "识别出的题目",
                "answer": "解答过程",
                "knowledge": "相关知识点",
            }
        """
        # ① 识别图片中的题目
        problem_text = self.extract_text(image_path)

        # ② 交给 Agent 解答（复用多 Agent 编排）
        from app.orchestrator_agent import OrchestratorAgent
        agent = OrchestratorAgent()

        # 先查教材找相关知识点
        search_prompt = "这道题涉及什么知识点？\n{}".format(problem_text)
        knowledge_result = agent.run(search_prompt, grade=grade, subject=subject)

        # 再解题
        solve_prompt = "请解答这道题，写出详细步骤：\n{}".format(problem_text)
        solution_result = agent.run(solve_prompt, grade=grade, subject=subject)

        return {
            "problem": problem_text,
            "answer": solution_result["answer"],
            "knowledge": knowledge_result["answer"],
        }

    def extract_text_from_base64(self, base64_image: str) -> str:
        """从 base64 编码的图片中提取文字

        Args:
            base64_image: base64 编码的图片数据

        Returns:
            str: 识别出的文本内容
        """
        response = self.client.chat.completions.create(
            model="qwen3-omni-flash",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请提取这张图片中的所有文字和数学公式，"
                                "保持原有格式和排版",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,{}".format(
                                base64_image
                            ),
                        },
                    },
                ],
            }],
            max_tokens=2048,
        )
        return response.choices[0].message.content
