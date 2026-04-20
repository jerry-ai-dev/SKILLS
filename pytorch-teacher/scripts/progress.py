"""
PyTorch 学习进度管理脚本
用法：
  python progress.py show          # 查看当前进度
  python progress.py complete <N>  # 标记第 N 课完成
  python progress.py reset <N>     # 重置第 N 课
  python progress.py reset-all     # 重置所有进度
"""
import json
import sys
import os
from datetime import datetime

PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "progress.json")

LESSONS = {
    1: "Tensor 张量入门",
    2: "自动求导 Autograd",
    3: "nn.Module 搭建神经网络",
    4: "训练循环 Training Loop",
    5: "数据加载 Dataset & DataLoader",
    6: "卷积神经网络 CNN",
    7: "序列模型与词嵌入",
    8: "Attention 注意力机制",
    9: "Transformer 架构",
    10: "从 Transformer 到 GPT",
    11: "预训练模型与 Hugging Face",
    12: "微调与 AI 前沿展望",
}

EXAMS = {
    "exam1": {"name": "📝 阶段考试 1: 基础篇", "after": 4, "covers": "Lesson 1-4"},
    "exam2": {"name": "📝 阶段考试 2: 实战进阶篇", "after": 7, "covers": "Lesson 5-7"},
    "exam3": {"name": "📝 阶段考试 3: Attention与Transformer", "after": 10, "covers": "Lesson 8-10"},
    "exam4": {"name": "🎓 期末综合考试", "after": 12, "covers": "Lesson 1-12"},
}

STAGES = {
    "第一阶段 基础": [1, 2, 3, 4, "exam1"],
    "第二阶段 实战进阶": [5, 6, 7, "exam2"],
    "第三阶段 Attention与Transformer": [8, 9, 10, "exam3"],
    "第四阶段 现代AI实践": [11, 12, "exam4"],
}


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"lessons": {}, "notes": [], "started_at": datetime.now().isoformat()}


def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def show_progress(progress):
    completed = sum(1 for v in progress["lessons"].values() if v.get("completed"))
    total = len(LESSONS) + len(EXAMS)
    bar_len = 24
    filled = int(bar_len * completed / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\n📊 PyTorch 学习进度: [{bar}] {completed}/{total}")
    print(f"   开始时间: {progress.get('started_at', 'N/A')}\n")

    for stage, items in STAGES.items():
        print(f"  {stage}:")
        for item in items:
            key = str(item)
            info = progress["lessons"].get(key, {})
            if isinstance(item, int):
                # Regular lesson
                status = "✅" if info.get("completed") else "⬜"
                extra = ""
                if info.get("completed_at"):
                    extra = f"  (完成于 {info['completed_at'][:10]})"
                if info.get("quiz_score") is not None:
                    extra += f"  测验: {info['quiz_score']}"
                print(f"    {status} Lesson {item}: {LESSONS[item]}{extra}")
            else:
                # Exam
                exam = EXAMS[item]
                status = "✅" if info.get("completed") else "⬜"
                extra = ""
                if info.get("completed_at"):
                    extra = f"  (完成于 {info['completed_at'][:10]})"
                if info.get("quiz_score") is not None:
                    extra += f"  成绩: {info['quiz_score']}分"
                print(f"    {status} {exam['name']}{extra}")
        print()

    # 下一课建议
    for stage, items in STAGES.items():
        for item in items:
            key = str(item)
            if not progress["lessons"].get(key, {}).get("completed"):
                if isinstance(item, int):
                    print(f"  👉 建议下一课: Lesson {item} - {LESSONS[item]}")
                else:
                    print(f"  👉 建议下一步: {EXAMS[item]['name']} (覆盖 {EXAMS[item]['covers']})")
                break
        else:
            continue
        break
    else:
        print("  🎓 恭喜！你已完成所有课程和考试！")

    if progress.get("notes"):
        print(f"\n  📝 学习笔记:")
        for note in progress["notes"][-5:]:
            print(f"    - {note}")


def complete_lesson(progress, lesson_id, quiz_score=None, note=None):
    key = str(lesson_id)
    if key not in progress["lessons"]:
        progress["lessons"][key] = {}
    progress["lessons"][key]["completed"] = True
    progress["lessons"][key]["completed_at"] = datetime.now().isoformat()
    if quiz_score is not None:
        progress["lessons"][key]["quiz_score"] = quiz_score
    if note:
        progress["notes"].append(f"[{key}] {note}")
    save_progress(progress)
    # Display name
    if key.startswith("exam"):
        name = EXAMS.get(key, {}).get("name", key)
    else:
        num = int(key)
        name = f"Lesson {num}: {LESSONS.get(num, '?')}"
    print(f"✅ {name} 已标记完成！")


def reset_lesson(progress, lesson_id):
    key = str(lesson_id)
    if key in progress["lessons"]:
        progress["lessons"][key] = {"completed": False}
        save_progress(progress)
    print(f"🔄 {key} 已重置")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    progress = load_progress()

    if cmd == "show":
        show_progress(progress)
    elif cmd == "complete":
        if len(sys.argv) < 3:
            print("用法: python progress.py complete <课程编号或exam1/exam2/exam3/exam4>")
            return
        lesson_id = sys.argv[2]
        # Support both numeric (1-12) and string (exam1-exam4) IDs
        if lesson_id.isdigit():
            lesson_id = int(lesson_id)
        score = sys.argv[3] if len(sys.argv) > 3 else None
        complete_lesson(progress, lesson_id, quiz_score=score)
        show_progress(progress)
    elif cmd == "reset":
        if len(sys.argv) < 3:
            print("用法: python progress.py reset <课程编号或exam1/exam2/exam3/exam4>")
            return
        lesson_id = sys.argv[2]
        if lesson_id.isdigit():
            lesson_id = int(lesson_id)
        reset_lesson(progress, lesson_id)
    elif cmd == "reset-all":
        save_progress({"lessons": {}, "notes": [], "started_at": datetime.now().isoformat()})
        print("🔄 所有进度已重置")
    else:
        print(f"未知命令: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
