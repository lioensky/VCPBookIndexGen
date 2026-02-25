"""
核心总结引擎：速读 / 精读双模式

速读(speed)：章节间并发，无上下文关联
精读(deep) ：章节间串行，携带滚动上下文
长章节     ：两种模式均采用步进式总结（总结的总结）
"""

import asyncio
import json
from typing import Optional
from openai import AsyncOpenAI
from .config import Config, count_tokens
from .extractor import Chapter

PROMPT_CHAPTER_SUMMARY = """你是客观中立的故事归纳程序。请对以下章节进行纯客观的结构化总结，不要包含任何主观评价、推断或读后感。

## 要求
1. 【剧情】客观陈述本章发生的实质性事件，按时间线梳理。
2. 【人物】客观列出出场人物及其具体行为。
3. 【设定】记录本章出现的新地点/物品/能力/法则等客观设定。
4. 【关系】客观记录人物之间实质性的交往与冲突。
5. 【关键】摘录本章的关键伏笔或悬念（只描述文本中明示的未解之谜，不要做扩展猜测）。

## 章节信息
- 标题：{title}

## 章节原文
{content}

请输出纯客观的结构化总结："""

PROMPT_PROGRESSIVE = """你是客观中立的故事归纳程序，正在归纳一个较长章节的第 {chunk_idx}/{total_chunks} 部分。

## 已读内容的阶段性总结
{partial_summary}

## 当前部分原文
{content}

请整合已读信息和当前内容，输出**更新后的本章客观总结**。不要包含任何评价或推测。保持结构：
1. 【剧情】…  2. 【人物】…  3. 【设定】…  4. 【关系】…  5. 【关键】…"""

PROMPT_DEEP_CHAPTER = """你是客观中立的故事归纳程序，正在进行带上文连贯记忆的深度归纳。

## 前文客观提要
{story_context}

## 当前章节
- 标题：{title}

{content}

请结合前文提要对本章进行纯客观的结构化总结，切忌任何形式的评价、心理揣测或上帝视角分析：
1. 【剧情】客观陈述本章发生的主要事件
2. 【人物】出场人物的具体行为动作
3. 【设定】客观补充新的世界观设定
4. 【关系】人物之间发生的实质交互
5. 【关键】客观记录新出现的悬念或前文悬念的揭晓"""

PROMPT_DEEP_PROGRESSIVE = """你是客观中立的故事归纳程序，正在归纳一个较长章节的第 {chunk_idx}/{total_chunks} 部分。

## 前文客观提要（之前章节累积）
{story_context}

## 本章已读部分的阶段性总结  
{partial_summary}

## 当前部分原文
{content}

请整合所有信息，输出更新后的完整章节客观总结（严禁包含主观评价）。"""

PROMPT_CONDENSE_CONTEXT = """请将以下故事阶段提要压缩为不超过 {max_tokens} tokens 的极简大纲，必须完全客观：
- 保留核心人物当前状态
- 保留推进中的主线事件
- 保留明确的未解悬念
- 保留核心环境设定
- 剔除所有的修饰词与感叹成分

当前提要：
{context}

压缩后的客观大纲："""

PROMPT_BOOK_SUMMARY = """以下是一本小说多章的客观剧情总结。请生成一份全书剧情最终概述。

## 要求（严禁使用文学评析、修辞赞美或主观读后感）
1. 客观还原故事的整体起因、经过、结果（3-5段）
2. 列举主要人物结局
3. 梳理核心事件链
4. 汇总关键世界观
5. 标明故事的客观转折节点

## 章节总结
{summaries}

请输出全书客观概述："""

PROMPT_MERGE_SUMMARIES = """以下是小说部分章节的客观总结（第 {start} 至 {end} 章）。
请将它们合并为一段纯客观的阶段性事件串联概述，禁止出现任何评注或观点。

{summaries}

阶段性客观概述："""


# ══════════════════════════════════════════════════
#  LLM 客户端
# ══════════════════════════════════════════════════

class LLMClient:
    """异步 LLM 调用封装（OpenAI 兼容接口）"""

    def __init__(self, config: Config):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    async def complete(self, prompt: str, max_retries: int = 3) -> str:
        """带重试的 LLM 调用"""
        for attempt in range(max_retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的小说内容分析与总结助手。"},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.config.summary_max_tokens,
                    temperature=self.config.temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** (attempt + 1)
                print(f"  ⚠ LLM 调用失败({e})，{wait}s 后重试...")
                await asyncio.sleep(wait)


# ══════════════════════════════════════════════════
#  步进式总结（长章节通用）
# ══════════════════════════════════════════════════

async def progressive_summarize(
    llm: LLMClient,
    chunks: list[str],
    title: str,
    story_context: str = "",  # 仅精读模式传入
) -> str:
    """
    对多个文本块进行步进式总结：
    chunk1 → summary1 → (summary1 + chunk2) → summary2 → ...
    """
    partial = ""
    is_deep = bool(story_context)
    total = len(chunks)

    for i, chunk in enumerate(chunks, 1):
        if i == 1 and not partial:
            # 首块
            if is_deep:
                prompt = PROMPT_DEEP_CHAPTER.format(
                    story_context=story_context, title=title, content=chunk
                )
            else:
                prompt = PROMPT_CHAPTER_SUMMARY.format(title=title, content=chunk)
        else:
            if is_deep:
                prompt = PROMPT_DEEP_PROGRESSIVE.format(
                    story_context=story_context,
                    partial_summary=partial,
                    content=chunk,
                    chunk_idx=i, total_chunks=total,
                )
            else:
                prompt = PROMPT_PROGRESSIVE.format(
                    partial_summary=partial,
                    content=chunk,
                    chunk_idx=i, total_chunks=total,
                )

        partial = await llm.complete(prompt)

    return partial


# ══════════════════════════════════════════════════
#  速读模式
# ══════════════════════════════════════════════════

class SpeedSummarizer:
    """并发速读：章节间无依赖，全部并发处理"""

    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMClient(config)
        self.semaphore = asyncio.Semaphore(config.max_concurrency)

    async def summarize_chapter(self, chapter: Chapter) -> str:
        """总结单个章节（自动判断是否需要步进）"""
        async with self.semaphore:
            content_tokens = chapter.token_count
            effective = self.config.effective_content_window

            if content_tokens <= effective:
                # 单次总结
                prompt = PROMPT_CHAPTER_SUMMARY.format(
                    title=chapter.title, content=chapter.content
                )
                return await self.llm.complete(prompt)
            else:
                # 步进式总结
                return await progressive_summarize(
                    self.llm, chapter.chunks, chapter.title
                )

    async def run(self, chapters: list[Chapter], on_done=None) -> list[Chapter]:
        """并发处理所有章节"""
        async def _process(ch: Chapter):
            ch.summary = await self.summarize_chapter(ch)
            if on_done:
                on_done(ch)
            return ch

        tasks = [_process(ch) for ch in chapters]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"  ✗ 第 {chapters[i].index} 章总结失败: {r}")
                chapters[i].summary = f"[总结失败: {r}]"

        return chapters


# ══════════════════════════════════════════════════
#  精读模式
# ══════════════════════════════════════════════════

class DeepSummarizer:
    """步进精读：携带滚动上下文，逐章阅读"""

    def __init__(self, config: Config, initial_context: str = ""):
        self.config = config
        self.llm = LLMClient(config)
        self.rolling_context = initial_context  # 滚动故事摘要

    async def _condense_context(self):
        """当滚动上下文超限时压缩"""
        ctx_tokens = count_tokens(self.rolling_context)
        if ctx_tokens > self.config.rolling_context_max:
            prompt = PROMPT_CONDENSE_CONTEXT.format(
                max_tokens=self.config.rolling_context_max // 2,
                context=self.rolling_context,
            )
            self.rolling_context = await self.llm.complete(prompt)

    async def summarize_chapter(self, chapter: Chapter) -> str:
        """带上下文总结单个章节"""
        content_tokens = chapter.token_count
        effective = self.config.effective_content_window
        # 减去上下文占用
        ctx_tokens = count_tokens(self.rolling_context)
        available = effective - ctx_tokens

        if available > 0 and content_tokens <= available:
            # 整章一次性总结
            prompt = PROMPT_DEEP_CHAPTER.format(
                story_context=self.rolling_context or "（这是故事的开端）",
                title=chapter.title,
                content=chapter.content,
            )
            return await self.llm.complete(prompt)
        else:
            # 步进式总结
            return await progressive_summarize(
                self.llm, chapter.chunks, chapter.title,
                story_context=self.rolling_context or "（这是故事的开端）",
            )

    async def run(self, chapters: list[Chapter], on_done=None) -> list[Chapter]:
        """顺序处理所有章节"""
        for ch in chapters:
            try:
                ch.summary = await self.summarize_chapter(ch)
                # 更新滚动上下文
                self.rolling_context += f"\n\n### 第{ch.index}章 {ch.title}\n{ch.summary}"
                await self._condense_context()
            except Exception as e:
                print(f"  ✗ 第 {ch.index} 章总结失败: {e}")
                ch.summary = f"[总结失败: {e}]"

            if on_done:
                on_done(ch, self.rolling_context)

        return chapters


# ══════════════════════════════════════════════════
#  全书总结生成器
# ══════════════════════════════════════════════════

class BookSummarizer:
    """基于所有章节总结生成全书总结（支持递归合并）"""

    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMClient(config)

    async def generate(self, chapters: list[Chapter]) -> str:
        """生成全书总结，自动处理超长情况"""
        summaries = [
            f"### 第{ch.index}章 {ch.title}\n{ch.summary}"
            for ch in chapters if ch.summary
        ]
        
        if not summaries:
            return "（无总结内容）"

        combined = "\n\n".join(summaries)
        combined_tokens = count_tokens(combined)
        effective = self.config.effective_content_window

        if combined_tokens <= effective:
            # 一次性生成
            prompt = PROMPT_BOOK_SUMMARY.format(summaries=combined)
            return await self.llm.complete(prompt)
        else:
            # 递归合并：分批 → 合并 → 再合并
            return await self._recursive_summarize(summaries)

    async def _recursive_summarize(self, summaries: list[str]) -> str:
        """递归分批合并总结"""
        effective = self.config.effective_content_window

        # 分批：每批尽量塞满有效窗口
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for s in summaries:
            st = count_tokens(s)
            if current_tokens + st > effective * 0.8 and current_batch:
                batches.append(current_batch)
                current_batch = [s]
                current_tokens = st
            else:
                current_batch.append(s)
                current_tokens += st
        if current_batch:
            batches.append(current_batch)

        # 合并每批
        merged: list[str] = []
        for i, batch in enumerate(batches):
            start = summaries.index(batch[0]) + 1
            end = summaries.index(batch[-1]) + 1
            print(f"  合并批次 {i+1}/{len(batches)} (约 {len(batch)} 章)...")
            prompt = PROMPT_MERGE_SUMMARIES.format(
                start=start, end=end,
                summaries="\n\n".join(batch),
            )
            result = await self.llm.complete(prompt)
            merged.append(result)

        # 如果合并后仍然超限，递归
        combined = "\n\n---\n\n".join(merged)
        if count_tokens(combined) > effective:
            return await self._recursive_summarize(merged)

        # 最终总结
        prompt = PROMPT_BOOK_SUMMARY.format(summaries=combined)
        return await self.llm.complete(prompt)
