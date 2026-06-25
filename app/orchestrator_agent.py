# orchestrator_agent.py
"""Orchestrator Agent（调度者）
负责：意图识别 → 任务拆解 → 工具调用 → 结果聚合

设计模式：Orchestrator + Specialist
- Orchestrator：负责任务拆解和调度（1 个）
- Specialist：负责具体执行（多个，各司其职）
"""

import inspect
import json
import re
from typing import List

from langchain_openai import ChatOpenAI
from app.config import settings
from app.agent_tools import (
    SearchTextbookTool, GenerateExamTool,
    AnalyzeStudyTool, CalculatorTool, ToolResult,
)


def _get_tool_param_names(tool) -> set:
    """获取工具 run() 方法的参数名集合（不含 self）"""
    sig = inspect.signature(tool.run)
    return {name for name in sig.parameters if name != "self"}


class OrchestratorAgent:
    """多 Agent 调度器"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.1,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_API_BASE,
        )
        # 注册所有 Specialist 工具
        self.tools = {
            "search_textbook": SearchTextbookTool(),
            "generate_exam": GenerateExamTool(),
            "analyze_study": AnalyzeStudyTool(),
            "calculator": CalculatorTool(),
        }
        # 预计算每个工具的有效参数名（用于安全注入）
        self._tool_params = {
            name: _get_tool_param_names(t)
            for name, t in self.tools.items()
        }

    def run(self, user_input: str,
            grade: str = "初三",
            subject: str = "数学",
            session_id: str = "") -> dict:
        """执行用户请求的主入口

        流程：
        ① 意图识别 → 判断用户想要什么
        ② 任务拆解 → 如果需要多步，拆成顺序任务
        ③ 逐步骤执行 → 上一步结果作为 context 传给下一步
        ④ 结果聚合 → 合并成自然语言回答
        """
        # ① 意图识别
        plan = self._plan(user_input, grade, subject)

        # ② 逐步骤执行（步骤间传递结果）
        results: List[ToolResult] = []
        step_outputs: List[str] = []  # 累积所有步骤的输出文本

        for i, step in enumerate(plan.get("steps", [])):
            tool_name = step.get("tool", "")
            tool = self.tools.get(tool_name)
            if not tool:
                results.append(ToolResult(
                    success=False,
                    error="未知工具: {}".format(tool_name),
                ))
                step_outputs.append("")
                continue

            params = dict(step.get("params", {}))

            # 如果不是第一步，注入前一步的结果作为 context
            if i > 0 and step_outputs:
                prev_context = "\n".join([
                    "步骤{}结果: {}".format(j + 1, out[:800])
                    for j, out in enumerate(step_outputs) if out
                ])
                if prev_context:
                    # 只注入到接受 context 参数的工具
                    if "context" in self._tool_params.get(tool_name, set()):
                        prev = params.get("context", "")
                        params["context"] = (
                            prev_context + "\n\n原始context: " + prev
                        )

            # 安全注入：只传工具 run() 方法实际接受的参数
            valid_params = self._tool_params.get(tool_name, set())
            safe_params = {}
            for k, v in params.items():
                if k in valid_params:
                    safe_params[k] = v

            # 补充默认的 grade / subject / session_id（仅当工具接受时）
            for extra_key in ["grade", "subject", "session_id"]:
                if extra_key in valid_params and extra_key not in safe_params:
                    val = locals().get(extra_key, "")
                    if val:
                        safe_params[extra_key] = val

            try:
                result = tool.run(**safe_params)
            except TypeError as e:
                result = ToolResult(
                    success=False,
                    error="参数错误 (工具={}): {}".format(tool_name, str(e)),
                )
            results.append(result)

            # 累积输出文本供下一步使用
            if result.success and result.data:
                output_text = str(result.data)
                # 如果是 search_textbook 返回的 dict，取 answer 字段
                if isinstance(result.data, dict) and "answer" in result.data:
                    output_text = result.data["answer"]
                step_outputs.append(output_text[:800])
            else:
                step_outputs.append("")

        # ③ 结果聚合
        final_answer = self._aggregate(user_input, plan, results)
        return {
            "answer": final_answer,
            "plan": plan,
            "steps": [
                {"tool": s.get("tool", ""), "success": r.success}
                for s, r in zip(plan.get("steps", []), results)
            ],
        }

    def _plan(self, user_input: str, grade: str, subject: str) -> dict:
        """用 LLM 判断用户意图，生成执行计划"""
        tools_desc = "\n".join([
            "- {}: {}".format(t.name, t.description)
            for t in self.tools.values()
        ])
        prompt = """你是一个教育助手的调度者。分析用户的问题，拆解成可执行的步骤。

可用工具：
{tools_desc}

用户问题：{user_input}
年级：{grade}
学科：{subject}

请输出 JSON 格式的执行计划，格式如下：
{{
    "reasoning": "简要分析",
    "steps": [
        {{"tool": "工具名", "params": {{"参数名": "值"}}}},
        ...
    ]
}}

只输出 JSON，不要解释：""".format(
            user_input=user_input, grade=grade,
            subject=subject, tools_desc=tools_desc,
        )

        response = self.llm.invoke(prompt)
        raw = response.content.strip()

        # 更健壮的 JSON 提取：去掉可能的 markdown 包裹和前缀文字
        # 先尝试去掉 ```json ... ``` 包裹
        json_candidate = raw
        code_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', raw, re.DOTALL)
        if code_match:
            json_candidate = code_match.group(1)
        else:
            # 找到最外层 { }
            brace_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if brace_match:
                json_candidate = brace_match.group(0)

        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass

        return {
            "reasoning": "无法解析 LLM 输出，使用默认问答",
            "steps": [{
                "tool": "search_textbook",
                "params": {"question": user_input,
                           "grade": grade, "subject": subject},
            }],
        }

    def _aggregate(self, user_input: str, plan: dict,
                   results: List[ToolResult]) -> str:
        """用 LLM 把多步结果合并成连贯的回答"""
        steps_lines = []
        for i, (s, r) in enumerate(zip(plan.get("steps", []), results)):
            if r.success:
                data_str = str(r.data)[:500]
            else:
                data_str = "[错误] {}".format(r.error)
            steps_lines.append("步骤{}: {} → {}".format(
                i + 1, s.get("tool", "?"), data_str
            ))
        steps_text = "\n".join(steps_lines)

        prompt = """你是教育助手。根据以下执行结果，给用户一个连贯的回答。

用户问题：{user_input}
执行过程：
{steps_text}

请用自然的语言回答用户，包含关键信息，必要时注明出处：""".format(
            user_input=user_input, steps_text=steps_text,
        )

        response = self.llm.invoke(prompt)
        return response.content
