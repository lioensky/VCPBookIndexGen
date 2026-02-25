"""配置管理与通用工具"""

import os
import re
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ── tiktoken 延迟加载 ──────────────────────────────
_encoder = None

def count_tokens(text: str) -> int:
    """计算文本 token 数（兼容无 tiktoken 环境）"""
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # 降级：中文约 1.5 token/字，英文约 1.3 token/词
            return int(len(text) * 0.6)
    return len(_encoder.encode(text))


def read_file(path: str, encoding: str = "auto") -> str:
    """读取文本文件，自动检测编码"""
    if encoding != "auto":
        with open(path, "r", encoding=encoding) as f:
            return f.read()
    # 按优先级尝试常见编码
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "big5", "latin-1"]:
        try:
            with open(path, "r", encoding=enc) as f:
                content = f.read()
            # 简单验证：非空且无大量替换字符
            if content and content.count("\ufffd") < len(content) * 0.01:
                return content
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法检测文件编码: {path}")


def sanitize_filename(name: str, max_len: int = 80) -> str:
    """清理文件名中的非法字符"""
    name = re.sub(r'[\\/:*?"<>|\r\n\t]', '', name)
    name = name.strip('. ')
    return name[:max_len] if len(name) > max_len else name


@dataclass
class Config:
    # LLM
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    max_context_tokens: int = 128_000
    summary_max_tokens: int = 2000
    temperature: float = 0.3

    # 分块
    chunk_size: int = 6000
    chunk_overlap: int = 200

    # 总结模式
    mode: str = "speed"
    max_concurrency: int = 5
    rolling_context_max: int = 4000

    # 文件
    file_encoding: str = "auto"
    output_dir: str = "./output"

    @classmethod
    def load(cls) -> "Config":
        return cls(
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            max_context_tokens=int(os.getenv("LLM_MAX_CONTEXT", "128000")),
            summary_max_tokens=int(os.getenv("SUMMARY_MAX_TOKENS", "2000")),
            temperature=float(os.getenv("TEMPERATURE", "0.3")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "6000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            mode=os.getenv("SUMMARY_MODE", "speed"),
            max_concurrency=int(os.getenv("MAX_CONCURRENCY", "5")),
            rolling_context_max=int(os.getenv("ROLLING_CONTEXT_MAX", "4000")),
            file_encoding=os.getenv("FILE_ENCODING", "auto"),
            output_dir=os.getenv("OUTPUT_DIR", "./output"),
        )

    @property
    def effective_content_window(self) -> int:
        """LLM 单次调用可用于内容的 token 数"""
        overhead = 800  # system prompt + 指令
        return self.max_context_tokens - overhead - self.summary_max_tokens
