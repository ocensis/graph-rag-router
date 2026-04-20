"""
HotpotQA 准备脚本（遵循 HippoRAG 评测协议）

1. 从 dev set 取前 50 个问题（先试水，后续可改成 1000）
2. 合并所有问题的 context 段落成语料库，写入 files/
3. 修改 settings.py 实体类型为通用英文
4. 生成评测用的 questions 和 answers 文件

用法：python prepare_hotpotqa.py [--count 50]
"""
import json
import argparse
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).parent
HOTPOT_FILE = PROJECT_ROOT / "datasets" / "hotpot_dev_distractor_v1.json"
FILES_DIR = PROJECT_ROOT / "files"
BACKUP_DIR = PROJECT_ROOT / "files_backup_medical"
SETTINGS_FILE = PROJECT_ROOT / "graphrag_agent" / "config" / "settings.py"
BENCH_DIR = PROJECT_ROOT / "bench_results"

parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=50, help="问题数量")
args = parser.parse_args()

# ============================================================
print("=" * 60)
print(f"HotpotQA 准备（取前 {args.count} 个问题）")
print("=" * 60)

# 加载数据
print("\n加载 HotpotQA dev set...")
with open(HOTPOT_FILE, "r", encoding="utf-8") as f:
    all_data = json.load(f)
print(f"  总问题数: {len(all_data)}")

# 取子集，bridge 和 comparison 各一半
bridge = [q for q in all_data if q["type"] == "bridge"]
comparison = [q for q in all_data if q["type"] == "comparison"]
half = args.count // 2
selected = bridge[:half] + comparison[:half]
selected = selected[:args.count]

types = {"bridge": sum(1 for q in selected if q["type"] == "bridge"),
         "comparison": sum(1 for q in selected if q["type"] == "comparison")}
print(f"  选取 {len(selected)} 个: {types}")

# ============================================================
print("\n" + "=" * 60)
print("第一步：备份当前 files/，写入 HotpotQA 语料")
print("=" * 60)

# 备份
if not BACKUP_DIR.exists():
    BACKUP_DIR.mkdir()
    moved = 0
    for item in FILES_DIR.iterdir():
        shutil.move(str(item), str(BACKUP_DIR / item.name))
        moved += 1
    print(f"  移走 {moved} 个文件到 {BACKUP_DIR.name}/")
else:
    for item in FILES_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    print(f"  备份已存在，清空 files/")

# 合并所有段落成语料库
seen_titles = set()
corpus_parts = []

for q in selected:
    for title, sentences in q["context"]:
        if title not in seen_titles:
            seen_titles.add(title)
            text = f"# {title}\n\n" + " ".join(sentences)
            corpus_parts.append((title, text))

print(f"  合并了 {len(corpus_parts)} 个不同段落")

# 每 10 个段落写一个 txt 文件
batch_size = 10
file_count = 0
for i in range(0, len(corpus_parts), batch_size):
    batch = corpus_parts[i:i + batch_size]
    content = "\n\n---\n\n".join([text for _, text in batch])
    filepath = FILES_DIR / f"hotpot_corpus_{file_count + 1:03d}.txt"
    filepath.write_text(content, encoding="utf-8")
    file_count += 1

print(f"  写入 {file_count} 个 txt 文件到 files/")

# ============================================================
print("\n" + "=" * 60)
print("第二步：生成评测文件")
print("=" * 60)

# 生成 bench 格式（和 run_graphrag_bench.py 兼容）
bench_questions = []
for i, q in enumerate(selected):
    bench_questions.append({
        "id": q["_id"],
        "question": q["question"],
        "question_type": q["type"],
        "answer": q["answer"],
        "supporting_facts": q["supporting_facts"],
    })

BENCH_DIR.mkdir(exist_ok=True)
q_file = BENCH_DIR / "hotpot_questions.json"
with open(q_file, "w", encoding="utf-8") as f:
    json.dump(bench_questions, f, ensure_ascii=False, indent=2)
print(f"  保存 {len(bench_questions)} 个问题到 {q_file.name}")

# ============================================================
print("\n" + "=" * 60)
print("第三步：修改 settings.py")
print("=" * 60)

settings_content = SETTINGS_FILE.read_text(encoding="utf-8")

old_kb = 'KB_NAME = "Medical"'
new_kb = 'KB_NAME = "HotpotQA"'
if old_kb in settings_content:
    settings_content = settings_content.replace(old_kb, new_kb)
    print(f"  KB_NAME -> HotpotQA")

old_theme = 'theme = "Medical knowledge and healthcare"'
new_theme = 'theme = "Wikipedia multi-hop QA"'
if old_theme in settings_content:
    settings_content = settings_content.replace(old_theme, new_theme)
    print(f"  theme -> Wikipedia multi-hop QA")

old_entity = '''entity_types = [
    "Disease",
    "Symptom",
    "Treatment",
    "Medication",
    "Body_Part",
    "Medical_Procedure",
    "Risk_Factor",
]'''

new_entity = '''entity_types = [
    "Person",
    "Organization",
    "Location",
    "Event",
    "Work",
    "Concept",
]'''

if old_entity in settings_content:
    settings_content = settings_content.replace(old_entity, new_entity)
    print(f"  entity_types -> 通用 Wikipedia 实体类型")

old_rel = '''relationship_types = [
    "treats",
    "causes",
    "diagnoses",
    "prevents",
    "symptom_of",
    "risk_factor_for",
    "interacts_with",
    "other",
]'''

new_rel = '''relationship_types = [
    "located_in",
    "member_of",
    "created_by",
    "part_of",
    "related_to",
    "occurred_in",
    "other",
]'''

if old_rel in settings_content:
    settings_content = settings_content.replace(old_rel, new_rel)
    print(f"  relationship_types -> 通用关系类型")

SETTINGS_FILE.write_text(settings_content, encoding="utf-8")

# ============================================================
print("\n" + "=" * 60)
print("准备完成！接下来：")
print("=" * 60)
print()
print("1. 清缓存:")
print("   rm -Recurse -Force cache/")
print()
print("2. 构建图谱:")
print("   python graphrag_agent/integrations/build/main.py")
print()
print("3. 跑评测（需要先写 run_hotpotqa_bench.py）")
