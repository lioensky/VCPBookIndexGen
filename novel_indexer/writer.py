"""索引文件输出 + 断点续传"""

import os
import json
from dataclasses import asdict
from .config import Config, sanitize_filename
from .extractor import Chapter, ChapterMark


class ProgressTracker:
    """断点续传管理"""

    def __init__(self, progress_path: str):
        self.path = progress_path
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "completed": [],
            "rolling_context": "",
            "mode": "",
        }

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def is_done(self, chapter_index: int) -> bool:
        return chapter_index in self.data["completed"]

    def mark_done(self, chapter_index: int, rolling_context: str = ""):
        """标记完成并同步保存当前的精读滚动上下文"""
        if chapter_index not in self.data["completed"]:
            self.data["completed"].append(chapter_index)
        
        # 实时持久化滚动上下文
        if rolling_context:
            self.data["rolling_context"] = rolling_context
            
        self.save()

    def get_rolling_context(self) -> str:
        """获取最后保存的精读推演上下文"""
        return self.data.get("rolling_context", "")

    @property
    def completed_count(self) -> int:
        return len(self.data["completed"])


class IndexWriter:
    """将处理结果写入索引文件夹"""

    def __init__(self, book_name: str, config: Config):
        self.book_name = book_name
        self.config = config
        safe_name = sanitize_filename(book_name)
        self.index_dir = os.path.join(config.output_dir, f"{safe_name}-索引")
        os.makedirs(self.index_dir, exist_ok=True)
        self.progress = ProgressTracker(
            os.path.join(self.index_dir, "progress.json")
        )

    # ── 章节目录 ──────────────────────────────────
    def write_toc(self, marks: list[ChapterMark]):
        """写入章节目录文件"""
        path = os.path.join(self.index_dir, f"{self.book_name}-章节目录.md")
        lines = [
            f"# {self.book_name} 章节目录\n",
            f"共 {len(marks)} 章\n",
            "---\n",
        ]
        for m in marks:
            lines.append(f"{m.index:>4d}. {m.title}\n")

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return path

    def read_toc(self) -> str:
        """读取已保存的章节目录"""
        path = os.path.join(self.index_dir, f"{self.book_name}-章节目录.md")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    # ── 章节文件 ──────────────────────────────────
    def write_chapter(self, chapter: Chapter):
        """写入单个章节的所有文件（原文与总结）"""
        prefix = sanitize_filename(f"{chapter.index:03d}-{chapter.title}")

        # 原文块
        for i, chunk in enumerate(chapter.chunks, 1):
            chunk_path = os.path.join(self.index_dir, f"{self.book_name}-{prefix}-原文块-{i}.md")
            if not os.path.exists(chunk_path):
                with open(chunk_path, "w", encoding="utf-8") as f:
                    f.write(f"# {chapter.title} - 原文块 {i}/{len(chapter.chunks)}\n\n")
                    f.write(chunk)

        # 总结
        if chapter.summary:
            summary_path = os.path.join(self.index_dir, f"{self.book_name}-{prefix}-总结.md")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(f"# {chapter.title} - 章节总结\n\n")
                f.write(chapter.summary)

    # ── 全书总结 ──────────────────────────────────
    def write_book_summary(self, summary: str):
        path = os.path.join(self.index_dir, f"{self.book_name}-全书剧情总结.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {self.book_name} - 全书剧情总结\n\n")
            f.write(summary)
        return path
