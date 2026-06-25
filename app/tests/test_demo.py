"""CI 验证用 —— 确保环境正常运行"""

def test_imports():
    """验证核心模块可导入"""
    from app.config import settings
    assert settings.APP_HOST is not None
    assert settings.DB_NAME == "edu_knowledge"


def test_health():
    """验证健康检查端点"""
    from app.edu_api import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_chat_model():
    """验证聊天请求模型"""
    from app.edu_api import EduChatRequest
    req = EduChatRequest(question="勾股定理是什么？")
    assert req.question == "勾股定理是什么？"
    assert req.grade == "初三"
    assert req.subject == "数学"
    assert req.session_id is None


def test_agent_model():
    """验证 Agent 请求模型"""
    from app.edu_api import AgentChatRequest
    req = AgentChatRequest(
        question="帮我查勾股定理再出3道题",
        session_id="test-session",
    )
    assert req.question == "帮我查勾股定理再出3道题"
    assert req.session_id == "test-session"


def test_tool_result():
    """验证 ToolResult 模型"""
    from app.agent_tools import ToolResult
    r = ToolResult(data={"answer": "test"})
    assert r.success is True
    assert r.data["answer"] == "test"

    r2 = ToolResult(success=False, error="出错了")
    assert r2.success is False
    assert "出错了" in r2.error
