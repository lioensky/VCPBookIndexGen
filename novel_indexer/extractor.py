"""
第一步：章节提取与文本分块

两阶段流程：
  1. 正则提取 → 预览 → 用户确认
  2. 基于确认的目录进行文本切分和分块
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from .config import count_tokens

# ══════════════════════════════════════════════════
#  数据模型
# ══════════════════════════════════════════════════

@dataclass
class ChapterMark:
    """章节标记（提取阶段）"""
    index: int              # 顺序编号 1-based
    title: str              # 完整标题行
    line_number: int        # 在原文中的行号
    char_offset: int        # 在原文中的字符偏移

@dataclass 
class Chapter:
    """完整章节（切分后）"""
    index: int
    title: str
    content: str            # 完整原文
    chunks: list[str] = field(default_factory=list)      # 分块后的原文
    summary: str = ""
    token_count: int = 0


# ══════════════════════════════════════════════════
#  正则模式库
# ══════════════════════════════════════════════════

_CN = '零一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟'

# (编译后正则, 模式名称, 优先级权重)
CHAPTER_PATTERNS: list[tuple[re.Pattern, str, float]] = [
    # ── 核心模式 ──
    (re.compile(rf'^第[{_CN}\d]+章[\s:：·—-]*(.*)', re.M),
     "第X章", 1.0),
    
    (re.compile(rf'^第[{_CN}\d]+[节][\s:：·—-]*(.*)', re.M),
     "第X节", 0.9),
    
    (re.compile(rf'^第[{_CN}\d]+[回][\s:：·—-]*(.*)', re.M),
     "第X回", 0.95),
    
    # ── 英文模式 ──
    (re.compile(r'^Chapter\s+(\d+)[\s:：.·—-]*(.*)', re.M | re.I),
     "Chapter N", 0.85),
    
    # ── 纯数字模式 ──
    (re.compile(r'^(\d{1,5})[\.、）)\s]\s*(.*\S)', re.M),
     "数字编号", 0.5),
]

# 特殊章节（序章/楔子/尾声等），独立匹配
SPECIAL_PATTERNS: list[re.Pattern] = [
    re.compile(r'^(序章|序言|序|楔子|引言|引子|前言|开篇)[\s:：·—-]*(.*)', re.M),
    re.compile(r'^(尾声|后记|终章|完结感言|番外\S{0,10}|大结局|epilogue)[\s:：·—-]*(.*)', re.M | re.I),
]


# ══════════════════════════════════════════════════
#  章节提取器
# ══════════════════════════════════════════════════

class ChapterExtractor:
    """从原始文本中提取章节标记"""

    def __init__(self, text: str):
        self.text = text
        self.lines = text.split('\n')
        # 预计算每行的字符偏移
        self._line_offsets = self._build_offsets()

    def _build_offsets(self) -> list[int]:
        offsets = []
        pos = 0
        for line in self.lines:
            offsets.append(pos)
            pos += len(line) + 1  # +1 for \n
        return offsets

    def auto_extract(self, custom_pattern: Optional[str] = None) -> list[ChapterMark]:
        """
        自动提取章节标记
        
        策略：尝试所有内置模式，选匹配数最多且得分最高的；
        再叠加特殊章节（序章/楔子/尾声等）
        """
        if custom_pattern:
            pat = re.compile(custom_pattern, re.M)
            marks = self._find_matches(pat)
            return self._build_marks(marks)

        # 尝试每种主要模式
        best_matches = []
        best_score = 0
        best_name = ""

        for pattern, name, weight in CHAPTER_PATTERNS:
            matches = self._find_matches(pattern)
            score = len(matches) * weight
            if score > best_score:
                best_score = score
                best_matches = matches
                best_name = name

        # 叠加特殊章节
        special = []
        for pat in SPECIAL_PATTERNS:
            special.extend(self._find_matches(pat))

        # 合并 + 去重（同行号去重）
        seen_lines = {m[0] for m in best_matches}
        for s in special:
            if s[0] not in seen_lines:
                best_matches.append(s)
                seen_lines.add(s[0])

        # 按行号排序
        best_matches.sort(key=lambda x: x[0])

        return self._build_marks(best_matches)

    def _find_matches(self, pattern: re.Pattern) -> list[tuple[int, str]]:
        """返回 [(行号, 标题文本), ...]"""
        matches = []
        for i, line in enumerate(self.lines):
            stripped = line.strip()
            if not stripped:
                continue
            m = pattern.match(stripped)
            if m:
                # 标题过长的跳过（可能是误匹配正文）
                if len(stripped) > 100:
                    continue
                matches.append((i, stripped))
        return matches

    def _build_marks(self, matches: list[tuple[int, str]]) -> list[ChapterMark]:
        return [
            ChapterMark(
                index=idx + 1,
                title=title,
                line_number=line_num,
                char_offset=self._line_offsets[line_num] if line_num < len(self._line_offsets) else 0,
            )
            for idx, (line_num, title) in enumerate(matches)
        ]

    def split_chapters(self, marks: list[ChapterMark]) -> list[Chapter]:
        """根据章节标记切分原文为 Chapter 列表"""
        chapters = []

        # ── 鲁棒性优化：捕获第一章之前的作品相关/前言内容 ──
        if marks and marks[0].char_offset > 0:
            preamble_text = self.text[:marks[0].char_offset].strip()
            # 过滤掉只有几个空格/换行的无意义内容
            if len(preamble_text) > 20: 
                chapters.append(Chapter(
                    index=0,
                    title="作品相关_前言",
                    content=preamble_text,
                    token_count=count_tokens(preamble_text)
                ))

        for i, mark in enumerate(marks):
            # 内容起点：标题行的下一行
            start = mark.char_offset + len(self.lines[mark.line_number]) + 1
            # 内容终点：下一个标记的偏移，或文末
            if i + 1 < len(marks):
                end = marks[i + 1].char_offset
            else:
                end = len(self.text)

            content = self.text[start:end].strip()
            tc = count_tokens(content) if content else 0

            chapters.append(Chapter(
                index=mark.index,
                title=mark.title,
                content=content,
                token_count=tc,
            ))
        return chapters


# ══════════════════════════════════════════════════
#  文本分块器
# ══════════════════════════════════════════════════

class TextChunker:
    """将文本按 token 上限分块，尽量在段落/句子边界切分"""

    def __init__(self, max_tokens: int = 6000, overlap_tokens: int = 200):
        self.max_tokens = max_tokens
        self.overlap = overlap_tokens

    def chunk(self, text: str) -> list[str]:
        """将文本切分为多个块"""
        total = count_tokens(text)
        if total <= self.max_tokens:
            if not text.strip():
                return []
            return [text]

        # 按段落切分
        paragraphs = self._split_paragraphs(text)
        
        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = count_tokens(para)

            # 超长段落：按句子二次切分
            if para_tokens > self.max_tokens:
                # 先保存当前累积
                if current_parts:
                    chunks.append('\n'.join(current_parts))
                    current_parts = []
                    current_tokens = 0
                # 句子级切分
                chunks.extend(self._chunk_by_sentences(para))
                continue

            if current_tokens + para_tokens > self.max_tokens and current_parts:
                chunk_text = '\n'.join(current_parts)
                chunks.append(chunk_text)
                
                # 重叠：保留尾部若干段落
                overlap_parts, overlap_tokens = self._get_overlap(current_parts)
                current_parts = overlap_parts + [para]
                current_tokens = overlap_tokens + para_tokens
            else:
                current_parts.append(para)
                current_tokens += para_tokens

        if current_parts:
            chunks.append('\n'.join(current_parts))

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """按空行或换行切段落"""
        # 先尝试双换行（标准段落分隔）
        parts = re.split(r'\n\s*\n', text)
        if len(parts) > 1:
            return [p.strip() for p in parts if p.strip()]
        # 退回单换行
        parts = text.split('\n')
        return [p.strip() for p in parts if p.strip()]

    def _chunk_by_sentences(self, text: str) -> list[str]:
        """按句子切分超长段落"""
        # 中文句子边界
        sentences = re.split(r'(?<=[。！？…\n；;])', text)
        sentences = [s for s in sentences if s.strip()]

        chunks = []
        current = ""
        current_tokens = 0

        for sent in sentences:
            st = count_tokens(sent)
            if current_tokens + st > self.max_tokens and current:
                chunks.append(current)
                # 简单重叠：保留最后一句
                current = sent
                current_tokens = st
            else:
                current += sent
                current_tokens += st

        if current:
            chunks.append(current)
        return chunks

    def _get_overlap(self, parts: list[str]) -> tuple[list[str], int]:
        """从段落列表尾部取出 overlap_tokens 的内容"""
        overlap_parts = []
        total = 0
        for p in reversed(parts):
            pt = count_tokens(p)
            if total + pt > self.overlap:
                break
            overlap_parts.insert(0, p)
            total += pt
        return overlap_parts, total
