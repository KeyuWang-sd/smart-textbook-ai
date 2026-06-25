-- 长期记忆模块数据库迁移脚本
-- 用于存储用户画像和错题记录

-- 1. 用户画像表
CREATE TABLE IF NOT EXISTS user_profiles (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    grade VARCHAR(50) NOT NULL,
    subject VARCHAR(50) NOT NULL,
    profile_data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (user_id, grade, subject)
);

-- 2. 错题记录表
CREATE TABLE IF NOT EXISTS wrong_answers (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    knowledge_point VARCHAR(500) NOT NULL,
    question TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 添加索引以加速查询
CREATE INDEX IF NOT EXISTS idx_user_profiles_user ON user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_wrong_answers_user ON wrong_answers(user_id);
CREATE INDEX IF NOT EXISTS idx_wrong_answers_kp ON wrong_answers(knowledge_point);
