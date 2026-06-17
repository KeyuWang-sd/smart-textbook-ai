# config.py
"""教育平台配置管理 —— 从环境变量 / .env 文件加载"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


class Settings:
    """应用配置"""

    # ---- 数据库 ----
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "edu_pass")
    DB_NAME: str = os.getenv("DB_NAME", "edu_knowledge")

    @property
    def database_url(self) -> str:
        return "postgresql://{user}:{password}@{host}:{port}/{db}".format(
            user=self.DB_USER,
            password=self.DB_PASSWORD,
            host=self.DB_HOST,
            port=self.DB_PORT,
            db=self.DB_NAME,
        )

    # ---- Redis ----
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # ---- LLM (DeepSeek) ----
    LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-v4-flash")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")

    # ---- Embedding (百炼 DashScope) ----
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

    # ---- 应用 ----
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    DATA_DIR: str = os.getenv("DATA_DIR", str(Path(__file__).parent.parent / "data"))


settings = Settings()
