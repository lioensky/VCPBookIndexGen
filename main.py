"""
长小说索引生成器 - 主入口

使用方式:
  python main.py <小说文件路径> [--name 书名] [--mode speed|deep] [--pattern 自定义正则]
"""

import sys
import os
import asyncio
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

from novel_indexer.config import Config, read_file, count_tokens
from novel_indexer.extractor import ChapterExtractor, TextChunker, Chapter
from novel_indexer.summarizer import SpeedSummarizer, DeepSummarizer, BookSummarizer
from novel_indexer.writer import IndexWriter

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="长小说索引生成器")
    parser.add_argument("file", help="小说文件路径 (.txt / .md)")
    parser.add_argument("--name", help="书名（默认取文件名）")
    parser.add_argument("--mode", choices=["speed", "deep"], help="覆盖 .env 中的总结模式")
    parser.add_argument("--pattern", help="自定义章节正则表达式")
    parser.add_argument("--skip-confirm", action="store_true", help="跳过目录确认（自动模式）")
    return parser.parse_args()


# ══════════════════════════════════════════════════
#  第一步：章节提取与确认
# ══════════════════════════════════════════════════

def step1_extract_chapters(text: str, book_name: str, config: Config,
                           custom_pattern: str = None, skip_confirm: bool = False):
    """
    交互式章节提取流程
    返回: (章节标记列表, 章节列表)
    """
    extractor = ChapterExtractor(text)

    while True:
        console.print("\n[bold cyan]═══ 第一步：章节目录提取 ═══[/]")

        marks = extractor.auto_extract(custom_pattern)

        if not marks:
            console.print("[red]⚠ 未检测到任何章节标记！[/]")
            action = Prompt.ask(
                "请选择操作",
                choices=["r", "m", "q"],
                default="r",
            )
            if action == "r":
                custom_pattern = Prompt.ask("请输入自定义正则表达式")
                continue
            elif action == "m":
                return _manual_toc_flow(book_name, config)
            else:
                sys.exit(0)

        # 预览目录
        _preview_toc(marks)
        
        console.print(f"\n[green]共检测到 {len(marks)} 个章节[/]")
        
        if skip_confirm:
            break

        action = Prompt.ask(
            "\n[bold]操作选择[/]: [y]确认 / [r]输入新正则重试 / [m]手动编辑目录文件 / [q]退出",
            choices=["y", "r", "m", "q"],
            default="y",
        )

        if action == "y":
            break
        elif action == "r":
            custom_pattern = Prompt.ask("请输入自定义正则表达式")
        elif action == "m":
            return _manual_toc_flow(book_name, config)
        else:
            sys.exit(0)

    # 切分章节
    chapters = extractor.split_chapters(marks)
    return marks, chapters


def _preview_toc(marks):
    """预览提取的章节目录"""
    table = Table(title="章节目录预览", show_lines=False)
    table.add_column("#", style="dim", width=6, justify="right")
    table.add_column("标题", style="cyan")
    table.add_column("行号", style="dim", justify="right")

    # 最多显示前20 + 后5
    display = marks[:20]
    if len(marks) > 25:
        display.append(None)  # 省略标记
        display.extend(marks[-5:])
    elif len(marks) > 20:
        display.extend(marks[20:])

    for m in display:
        if m is None:
            table.add_row("...", f"... 省略 {len(marks) - 25} 章 ...", "...")
        else:
            table.add_row(str(m.index), m.title, str(m.line_number))

    console.print(table)


def _manual_toc_flow(book_name, config):
    """手动编辑目录文件流程"""
    writer = IndexWriter(book_name, config)
    toc_path = os.path.join(writer.index_dir, f"{book_name}-章节目录.md")

    # 生成模板
    if not os.path.exists(toc_path):
        with open(toc_path, "w", encoding="utf-8") as f:
            f.write(f"# {book_name} 章节目录\n\n")
            f.write("# 请每行写一个章节标题，格式：\n")
            f.write("# 第一章 标题\n")
            f.write("# 第二章 标题\n")
            f.write("# ...\n")

    console.print(f"\n[yellow]请手动编辑目录文件后重新运行！逻辑待后续完善。[/]\n  {toc_path}")
    sys.exit(0)


# ══════════════════════════════════════════════════
#  第二步：分块 + 总结
# ══════════════════════════════════════════════════

async def step2_process(chapters: list[Chapter], book_name: str, config: Config):
    """分块 → 总结 → 写入文件"""
    writer = IndexWriter(book_name, config)
    chunker = TextChunker(config.chunk_size, config.chunk_overlap)
    
    # ── 分块 ──
    console.print("\n[bold cyan]═══ 文本分块 ═══[/]")
    total_chunks = 0
    for ch in chapters:
        ch.chunks = chunker.chunk(ch.content)
        total_chunks += len(ch.chunks)

    console.print(
        f"  共 {len(chapters)} 章，{total_chunks} 个文本块，"
        f"每块上限 {config.chunk_size} tokens"
    )

    # ── 过滤未总结章节 ──
    todo = []
    for ch in chapters:
        if not writer.progress.is_done(ch.index):
            todo.append(ch)
        else:
            # 对于深读模式的断点续传，即使章节跳过，我们也需要在 DeepSummarizer 初始化时传入最后的 rolling_context
            pass
    
    if todo:
        done_count = len(chapters) - len(todo)
        if done_count > 0:
            console.print(f"\n[yellow]断点续传：跳过已完成的 {done_count} 章[/]")
        
        console.print(f"\n[bold cyan]═══ 章节总结 ({config.mode}模式) ═══[/]")
        
        completed = done_count

        def on_chapter_done(ch: Chapter, rolling_context: str = ""):
            nonlocal completed
            completed += 1
            # 写入总结和原文块
            writer.write_chapter(ch)
            # 标记进度并持久化上下文
            writer.progress.mark_done(ch.index, rolling_context)
            console.print(f"  [{completed}/{len(chapters)}] ✓ {ch.title}")

        if config.mode == "speed":
            summarizer = SpeedSummarizer(config)
            # 速读无滚动上下文
            await summarizer.run(todo, on_done=lambda ch: on_chapter_done(ch, ""))
        else:
            # 精读：读取上一次保存的 rolling_context
            last_context = writer.progress.get_rolling_context()
            if last_context:
                console.print("[dim]  已恢复上次的精读上下文...[/]")
            summarizer = DeepSummarizer(config, initial_context=last_context)
            await summarizer.run(todo, on_done=on_chapter_done)
    else:
        console.print("\n[green]所有章节总结已完成（从断点恢复）[/]")
    
    # 因为生成全书总结需要每一章的 summary 内容，我们需要重新加载它们。
    # 这里我们简化处理：假设文件已被持久化，我们通过文件反读：
    _load_summaries_from_disk(chapters, writer)

    # ── 全书总结 ──
    console.print("\n[bold cyan]═══ 生成全书总结 ═══[/]")
    book_summarizer = BookSummarizer(config)
    book_summary = await book_summarizer.generate(chapters)
    summary_path = writer.write_book_summary(book_summary)
    console.print(f"  ✓ 全书总结已写入: {summary_path}")

    # ── 写入最终目录 ──
    console.print(f"\n[bold green]═══ 完成！索引目录: {writer.index_dir} ═══[/]")

def _load_summaries_from_disk(chapters: list[Chapter], writer: IndexWriter):
    """从磁盘补回已生成的 summary 内容"""
    from novel_indexer.config import sanitize_filename
    for ch in chapters:
        if not ch.summary:
            prefix = sanitize_filename(f"{ch.index:03d}-{ch.title}")
            summary_path = os.path.join(writer.index_dir, f"{writer.book_name}-{prefix}-总结.md")
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # 去掉我们自己加在头部的标题 "# {title} - 章节总结\n\n"
                    header_marker = f"# {ch.title} - 章节总结\n\n"
                    if content.startswith(header_marker):
                        content = content[len(header_marker):]
                    ch.summary = content


# ══════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════

def main():
    args = parse_args()
    config = Config.load()

    if args.mode:
        config.mode = args.mode

    book_name = args.name or os.path.splitext(os.path.basename(args.file))[0]

    console.print(Panel.fit(
        f"[bold]{book_name}[/]\n"
        f"文件: {args.file}\n"
        f"模式: [cyan]{config.mode}[/] | 模型: [cyan]{config.model}[/]\n"
        f"分块: {config.chunk_size} tokens | 并发: {config.max_concurrency}",
        title="长小说索引生成器",
        border_style="bright_blue",
    ))

    console.print("\n[dim]读取文件中...[/]")
    text = read_file(args.file, config.file_encoding)
    total_chars = len(text)
    total_tokens = count_tokens(text)
    console.print(
        f"  文件大小: {total_chars:,} 字符 ≈ {total_tokens:,} tokens "
        f"({os.path.getsize(args.file) / 1024 / 1024:.1f} MB)"
    )

    marks, chapters = step1_extract_chapters(
        text, book_name, config,
        custom_pattern=args.pattern,
        skip_confirm=args.skip_confirm,
    )

    writer = IndexWriter(book_name, config)
    toc_path = writer.write_toc(marks)
    console.print(f"\n  ✓ 章节目录已写入: {toc_path}")

    asyncio.run(step2_process(chapters, book_name, config))


if __name__ == "__main__":
    main()
