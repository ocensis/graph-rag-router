"""
一键全链路重建管道 —— 切模型后从 0 到完整 KG

执行顺序（有依赖，必须按序）:
  Step 1: scripts/build/nuke_and_rebuild.py          删 Neo4j + chunker + embedding + 实体抽取 + 初版社区摘要（40-60min）
  Step 2: scripts/maintenance/dedup_case_variants.py case-variant 实体合并兜底（1-2min）
  Step 3: scripts/build/rebuild_hierarchical_summaries.py 分层摘要 level 0/2/4（15-25min）
  Step 4: scripts/eval/graph_quality_report.py       KG 质量报告 + LLM judge 采样（1-2min）

前置条件:
  - .env 已手动切到目标模型（如 google/gemini-3-flash-preview）
  - docker compose 能正常启动 Neo4j
  - files/rag_papers/ 目录下有 PDF

用法:
  python _full_rebuild_pipeline.py                        # 全跑
  python _full_rebuild_pipeline.py --skip-nuke            # 跳重建，只跑 2+3+4
  python _full_rebuild_pipeline.py --skip-summaries       # 跳分层摘要
  python _full_rebuild_pipeline.py --continue-on-error    # 某步失败继续下一步（默认 fail-fast）
  python _full_rebuild_pipeline.py --log /tmp/rebuild.log # 写入日志文件
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# 脚本在 scripts/build/ 下，项目根目录是上 2 层
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PYTHON = sys.executable


def banner(msg: str, border_char: str = "="):
    """打个醒目的横幅"""
    line = border_char * 72
    print(f"\n{line}")
    print(f"  {msg}")
    print(f"{line}\n")


def run_step(name: str, cmd: list, continue_on_error: bool, log_fh=None) -> tuple[bool, float]:
    """
    跑一个 step，返回 (success, elapsed_seconds)
    子进程 stdout/stderr 实时转发到本进程 + 可选 log 文件

    Windows 下子进程 stdout=PIPE 时 Python 默认用系统编码（中文 Windows = cp936）
    写，导致父进程按 UTF-8 读时乱码。必须同时强制子进程也用 UTF-8：
      - PYTHONUTF8=1: Python UTF-8 模式（PEP 540），影响 open() / sys.std* 所有 I/O
      - PYTHONIOENCODING=utf-8: 显式指定 stdin/stdout/stderr 编码
    """
    banner(f"[{name}] START  |  cmd: {' '.join(cmd)}")
    start = time.time()

    child_env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }

    # 关键：把子进程 stdout 实时接管并回显
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=child_env,
    )
    assert proc.stdout is not None

    try:
        for line in proc.stdout:
            print(line, end="")
            if log_fh:
                log_fh.write(line)
                log_fh.flush()
    except KeyboardInterrupt:
        proc.terminate()
        banner(f"[{name}] KeyboardInterrupt, 子进程已终止", "!")
        return False, time.time() - start

    ret = proc.wait()
    elapsed = time.time() - start

    if ret == 0:
        banner(f"[{name}] OK  |  耗时 {elapsed/60:.1f} 分钟", "-")
        return True, elapsed
    else:
        banner(f"[{name}] FAILED  |  exit={ret}  |  耗时 {elapsed/60:.1f} 分钟", "!")
        if not continue_on_error:
            print(f"→ fail-fast: 后续 step 被跳过（用 --continue-on-error 覆盖）")
        return False, elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-nuke", action="store_true", help="跳过 Step 1（适用于仅重跑摘要）")
    ap.add_argument("--skip-dedup", action="store_true", help="跳过 Step 2")
    ap.add_argument("--skip-summaries", action="store_true", help="跳过 Step 3")
    ap.add_argument("--skip-quality", action="store_true", help="跳过 Step 4")
    ap.add_argument("--continue-on-error", action="store_true", help="某步失败仍跑下一步")
    ap.add_argument("--workers", type=int, default=12, help="摘要并发数（Step 3）")
    ap.add_argument("--sample", type=int, default=30, help="LLM judge 采样数（Step 4）")
    ap.add_argument("--log", type=str, default=None, help="日志文件路径（除 stdout 之外再写一份）")
    args = ap.parse_args()

    # 顶层横幅
    banner(f"GraphRAG 全链路重建管道  |  {datetime.now().isoformat(timespec='seconds')}")

    # 显示关键环境
    print(f"Python:         {PYTHON}")
    print(f"Project root:   {PROJECT_ROOT}")
    print(f"Env LLM model:  {os.getenv('OPENAI_LLM_MODEL', '(未设置)')}")
    print(f"Env base_url:   {os.getenv('OPENAI_BASE_URL', '(未设置)')}")
    print(f"Continue on err:{args.continue_on_error}")
    print(f"Log file:       {args.log or '(无)'}\n")

    # 打开日志文件
    log_fh = None
    if args.log:
        log_fh = open(args.log, "w", encoding="utf-8", buffering=1)
        log_fh.write(f"Started {datetime.now().isoformat(timespec='seconds')}\n")

    # 构造 step 列表
    steps = []
    if not args.skip_nuke:
        steps.append((
            "Step 1/4  Nuke + Rebuild (chunking + embedding + 实体抽取)",
            [PYTHON, str(SCRIPTS_DIR / "build" / "nuke_and_rebuild.py"), "--yes"],
        ))
    if not args.skip_dedup:
        steps.append((
            "Step 2/4  Case-variant Dedup",
            [PYTHON, str(SCRIPTS_DIR / "maintenance" / "dedup_case_variants.py")],
        ))
    if not args.skip_summaries:
        steps.append((
            "Step 3/4  Hierarchical Community Summaries (level 0/2/4)",
            [PYTHON, str(SCRIPTS_DIR / "build" / "rebuild_hierarchical_summaries.py"),
             "--workers", str(args.workers)],
        ))
    if not args.skip_quality:
        steps.append((
            "Step 4/4  KG Quality Report (+LLM judge 采样)",
            [
                PYTHON, str(SCRIPTS_DIR / "eval" / "graph_quality_report.py"),
                "--sample", str(args.sample),
                "--out", "benchmarks/results/kg_quality_latest.md",
            ],
        ))

    if not steps:
        print("所有 step 都被 skip，无事可做")
        return

    print(f"将执行 {len(steps)} 个 step:")
    for name, cmd in steps:
        print(f"  - {name}")
    print()

    # 顺序跑
    pipeline_start = time.time()
    results = []
    for name, cmd in steps:
        ok, elapsed = run_step(name, cmd, args.continue_on_error, log_fh)
        results.append((name, ok, elapsed))
        if not ok and not args.continue_on_error:
            break

    # 总结
    total = time.time() - pipeline_start
    banner(f"管道结束  |  总耗时 {total/60:.1f} 分钟")
    print(f"{'Step':<60s} {'Status':<8s} {'Time':>8s}")
    print("-" * 80)
    for name, ok, elapsed in results:
        status = "OK" if ok else "FAIL"
        print(f"{name[:60]:<60s} {status:<8s} {elapsed/60:>6.1f}m")

    all_ok = all(ok for _, ok, _ in results) and len(results) == len(steps)
    if all_ok:
        print("\n全部完成。下一步建议:")
        print("  python run_hotpotqa_bench.py --agent router "
              "--questions bench_results/eng_cross_doc_questions.json "
              "--tag eng200_new --workers 8")

    if log_fh:
        log_fh.write(f"\nEnded {datetime.now().isoformat(timespec='seconds')}, total {total/60:.1f}min\n")
        log_fh.close()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
