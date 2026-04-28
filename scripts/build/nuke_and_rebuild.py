"""
一键重置 Neo4j + 切 rag_papers + 重建全图

步骤:
1. docker compose down -v (删数据卷)
2. docker compose up -d (重启)
3. 等 Neo4j healthy (polling)
4. 切 KB 到 rag_papers
5. 跑 ingestion pipeline (最慢，40-60 分钟)

中间任一步失败都会停止，不会让你一路跑错。

用法:
  python _nuke_and_rebuild.py               # 完整流程（交互确认）
  python _nuke_and_rebuild.py --yes         # 非交互，直接跑
  python _nuke_and_rebuild.py --no-build    # 只重置不建图（调试用）
  python _nuke_and_rebuild.py --yes 2>&1 | tee /tmp/rebuild.log   # 写日志
"""
import subprocess
import time
import sys
import os
import argparse
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# 脚本在 scripts/build/ 下，项目根是上 2 层
PROJECT_ROOT = Path(__file__).parent.parent.parent


def run(cmd, check=True, capture=False):
    """执行命令，失败就打印 + 抛错"""
    print(f"\n>>> {cmd}")
    if capture:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result
    else:
        return subprocess.run(cmd, shell=True, check=check)


def wait_for_neo4j(max_wait=120):
    """轮询 Neo4j 直到 healthy"""
    print(f"\n等 Neo4j 启动 ...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            result = subprocess.run(
                'docker exec neo4j-graphrag-clean cypher-shell -u neo4j -p 12345678 "RETURN 1"',
                shell=True, capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                print(f"  ✓ Neo4j 就绪 ({time.time()-start:.0f}s)")
                return True
        except Exception:
            pass
        print(f"  ...等待中 ({time.time()-start:.0f}s)")
        time.sleep(5)
    print(f"  ✗ Neo4j 超时 {max_wait}s 未就绪")
    return False


def switch_kb(kb_name):
    """修改 .env 的 KB_NAME"""
    env_file = PROJECT_ROOT / ".env"
    content = env_file.read_text(encoding="utf-8")
    import re
    if re.search(r'^KB_NAME\s*=', content, re.MULTILINE):
        content = re.sub(r'^KB_NAME\s*=.*$', f"KB_NAME = '{kb_name}'", content, flags=re.MULTILINE)
    else:
        content = f"KB_NAME = '{kb_name}'\n" + content
    env_file.write_text(content, encoding="utf-8")
    print(f"  ✓ KB_NAME 切换到 {kb_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-build", action="store_true", help="只重置 Neo4j 不建图")
    parser.add_argument("--kb", type=str, default="rag_papers", help="目标 KB 名字")
    parser.add_argument("--yes", action="store_true", help="跳过所有交互确认（用于后台运行）")
    args = parser.parse_args()

    pipeline_start = time.time()

    # 用户确认
    print("=" * 60)
    print("⚠  这会删除 Neo4j 所有数据（包括 HotpotQA / mini_wiki / rag_papers）")
    print("=" * 60)
    if not args.yes:
        confirm = input("\n继续？输入 YES 确认: ").strip()
        if confirm != "YES":
            print("已取消")
            sys.exit(0)
    else:
        print("--yes 模式，跳过确认")

    os.chdir(PROJECT_ROOT)

    # Step 1: 停并删卷
    print(f"\n[1/5] 停 Neo4j + 删数据卷 ...")
    run("docker compose down -v")

    # Step 2: 启动
    print(f"\n[2/5] 启动 Neo4j ...")
    run("docker compose up -d")

    # Step 3: 等 healthy
    print(f"\n[3/5] 等 Neo4j ready ...")
    if not wait_for_neo4j(120):
        print("Neo4j 启动失败，检查 docker logs")
        sys.exit(1)

    # 验证是空的
    print(f"\n验证 Neo4j 是空的:")
    run('docker exec neo4j-graphrag-clean cypher-shell -u neo4j -p 12345678 "MATCH (n) RETURN count(n) AS total"')

    # Step 4: 切 KB
    print(f"\n[4/5] 切 KB 到 {args.kb} ...")
    switch_kb(args.kb)

    # 验证 FILES_DIR 指向正确的子目录（settings.py 已按 KB 隔离）
    from graphrag_agent.config.settings import FILES_DIR
    print(f"\n当前 FILES_DIR = {FILES_DIR}")
    if FILES_DIR.exists():
        file_count = sum(1 for _ in FILES_DIR.rglob("*") if _.is_file())
        print(f"  文件数: {file_count}")
    else:
        print(f"  ⚠ 目录不存在！请确认 files/{args.kb}/ 有文件")
        sys.exit(1)

    if not args.no_build and not args.yes:
        ans = input(f"\n继续建图？ (y/N): ").strip().lower()
        if ans != "y":
            print("已取消建图步骤。手动继续：python graphrag_agent/integrations/build/main.py")
            sys.exit(0)

    # Step 5: 建图
    if args.no_build:
        print(f"\n[5/5] 跳过建图 (--no-build)")
        print(f"\n手动继续: python graphrag_agent/integrations/build/main.py")
    else:
        print(f"\n[5/5] 跑 ingestion pipeline （40-60 分钟，去冲杯咖啡）...")
        t0 = time.time()
        run(f"{sys.executable} graphrag_agent/integrations/build/main.py")
        print(f"\n  ✓ 建图完成 ({(time.time()-t0)/60:.1f} 分钟)")

    total_min = (time.time() - pipeline_start) / 60
    print("\n" + "=" * 60)
    print(f"全部完成! 总耗时 {total_min:.1f} 分钟")
    print("=" * 60)
    print("\n下一步建议:")
    print("  1. 跑 KG 质量报告:   python _graph_quality_report.py --sample 30")
    print("  2. 跑 dedup 验证:    python _dedup_case_variants.py --dry-run")
    print("  3. 跑 router bench:  python run_hotpotqa_bench.py --agent router "
          "--questions bench_results/eng_cross_doc_questions.json --tag eng200_dehyphen --workers 8")


if __name__ == "__main__":
    main()
