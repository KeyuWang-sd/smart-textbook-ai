# memory_store.py
"""记忆存储层（Memory Store）
双层架构：Redis 短期记忆 + pgvector 长期记忆

短期记忆 → 当前会话上下文（缓存，TTL 1 小时）
长期记忆 → 用户的错题记录、薄弱知识点（持久化）
"""

import asyncio
import json
from typing import List, Optional
from datetime import datetime

from app.config import settings


class ShortTermMemory:
    """短期记忆：会话内上下文（Redis，TTL 1 小时）"""

    def __init__(self):
        import redis.asyncio as aioredis
        self.redis = aioredis.from_url(settings.REDIS_URL)

    async def save_context(self, session_id: str,
                           user_msg: str, agent_msg: str,
                           sources: Optional[list] = None):
        """保存一轮对话"""
        entry = {
            "user": user_msg,
            "agent": agent_msg,
            "sources": sources or [],
            "timestamp": datetime.now().isoformat()
        }
        await self.redis.rpush(f"chat:{session_id}", json.dumps(entry))
        await self.redis.expire(f"chat:{session_id}", 3600)  # 1 小时过期

    async def get_history(self, session_id: str,
                          last_n: int = 5) -> List[dict]:
        """获取最近 N 轮对话"""
        raw = await self.redis.lrange(f"chat:{session_id}", -last_n, -1)
        return [json.loads(r) for r in raw]

    async def build_prompt_context(self, session_id: str) -> str:
        """构建带上下文的 prompt 片段"""
        history = await self.get_history(session_id, last_n=5)
        if not history:
            return ""
        lines = ["\n[对话历史]"]
        for h in history:
            lines.append(f"用户: {h['user']}")
            lines.append(f"助手: {h['agent']}")
        return "\n".join(lines)

    async def clear_history(self, session_id: str):
        """清除指定会话的短期记忆"""
        await self.redis.delete(f"chat:{session_id}")


class LongTermMemory:
    """长期记忆：用户画像 + 薄弱知识点（pgvector + PostgreSQL）

    注意：所有数据库操作在线程池执行，不阻塞 FastAPI 事件循环。
    """

    def __init__(self):
        self.connection = settings.database_url

    def _connect(self):
        """同步数据库连接（仅在线程池内调用）"""
        import psycopg2
        return psycopg2.connect(self.connection)

    # ---- 内部同步方法（在线程池执行） ----

    def _save_user_profile_sync(self, user_id: str, profile: dict):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO user_profiles (user_id, grade, subject, profile_data, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (user_id, grade, subject)
                DO UPDATE SET profile_data = %s, updated_at = NOW()
            """, (
                user_id,
                profile.get("grade", ""),
                profile.get("subject", ""),
                json.dumps(profile),
                json.dumps(profile),
            ))
            conn.commit()
        finally:
            conn.close()

    def _get_user_profile_sync(
        self, user_id: str, grade: str,
        subject: str,
    ) -> Optional[dict]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT profile_data FROM user_profiles
                WHERE user_id = %s AND grade = %s AND subject = %s
            """, (user_id, grade, subject))
            row = cur.fetchone()
            if row:
                return json.loads(row[0])
            return None
        finally:
            conn.close()

    def _record_wrong_answer_sync(
        self, user_id: str, knowledge_point: str, question: str,
    ):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO wrong_answers (user_id, knowledge_point, question, created_at)
                VALUES (%s, %s, %s, NOW())
            """, (user_id, knowledge_point, question))
            conn.commit()
        finally:
            conn.close()

    # ---- 公开异步接口 ----

    async def save_user_profile(self, user_id: str, profile: dict):
        """保存或更新用户画像

        profile 字段:
            - grade: 年级
            - subject: 学科
            - weak_points: list[str] 薄弱知识点
            - mastered_kps: list[str] 已掌握知识点
        """
        await asyncio.to_thread(
            self._save_user_profile_sync, user_id, profile
        )

    async def get_user_profile(self, user_id: str, grade: str = "",
                               subject: str = "") -> Optional[dict]:
        """获取用户画像"""
        return await asyncio.to_thread(
            self._get_user_profile_sync, user_id, grade, subject
        )

    async def get_weak_points(self, user_id: str) -> list:
        """获取用户薄弱知识点"""
        profile = await self.get_user_profile(user_id)
        if profile:
            return profile.get("weak_points", [])
        return []

    async def record_wrong_answer(
        self, user_id: str, knowledge_point: str, question: str,
    ):
        """记录错题

        存入 pgvector 的 wrong_answers 表，后续可检索相似错题
        """
        await asyncio.to_thread(
            self._record_wrong_answer_sync, user_id, knowledge_point, question
        )
