"""
后训练理论深化 - 学习进度管理脚本
用法：
  python progress.py show                       # 查看当前进度
  python progress.py complete <N>               # 标记第 N 课完成（N=1-10）
  python progress.py complete exam<N> [score]  # 标记第 N 次考试完成，可附分数
  python progress.py reset <N>                  # 重置第 N 课
  python progress.py reset-all                  # 重置所有进度
"""
import json
import sys
import os
from datetime import datetime

PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "progress.json")

LESSONS = {
    1:  "强化学习基础 & MDP",
    2:  "Policy Gradient & REINFORCE",
    3:  "PPO 算法深入",
    4:  "Reward Model 奖励模型",
    5:  "RLHF 完整流程",
    6:  "GRPO 算法",
    7:  "SFT 工程实践",
    8:  "梯度累积 & 混合精度训练",
    9:  "BPE Tokenizer 原理",
    10: "DeepSeek R1 论文精读",
}

EXAMS = {
    "exam1": {"name": "📝 阶段考试 1: RL 理论基础", "after": 3, "covers": "Lesson 1-3"},
    "exam2": {"name": "📝 阶段考试 2: RLHF 完整流程", "after": 7, "covers": "Lesson 4-7"},
    "exam3": {"name": "🎓 期末综合考试", "after": 10, "covers": "Lesson 1-10"},
}

STAGES = {
    "第一阶段 RL 理论基础":    [1, 2, 3, "exam1"],
    "第二阶段 RLHF 完整流程":  [4, 5, 6, 7, "exam2"],
    "第三阶段 工程与前沿":      [8, 9, 10, "exam3"],
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
    bar_len = 26
    filled = int(bar_len * completed / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\n📊 后训练理论深化进度: [{bar}] {completed}/{total}")
    print(f"   开始时间: {progress.get('started_at', 'N/A')}\n")

    for stage, items in STAGES.items():
        print(f"  {stage}:")
        for item in items:
            key = str(item)
            info = progress["lessons"].get(key, {})
            if isinstance(item, int):
                status = "✅" if info.get("completed") else "⬜"
                extra = ""
                if info.get("completed_at"):
                    extra = f"  (完成于 {info['completed_at'][:10]})"
                print(f"    {status} Lesson {item}: {LESSONS[item]}{extra}")
            else:
                exam_info = EXAMS[item]
                status = "✅" if info.get("completed") else "⬜"
                extra = ""
                if info.get("score") is not None:
                    grade = get_grade(info["score"])
                    extra = f"  得分: {info['score']}/100  ({grade})"
                if info.get("completed_at"):
                    extra += f"  (完成于 {info['completed_at'][:10]})"
                print(f"    {status} {exam_info['name']}  [{exam_info['covers']}]{extra}")
        print()


def get_grade(score):
    if score >= 90: return "优秀 🏆"
    if score >= 75: return "良好 ✨"
    if score >= 60: return "及格 👍"
    return "需复习 📖"


def complete_lesson(progress, key_str):
    if key_str not in progress["lessons"]:
        progress["lessons"][key_str] = {}
    progress["lessons"][key_str]["completed"] = True
    progress["lessons"][key_str]["completed_at"] = datetime.now().isoformat()

    # 判断是课程还是考试
    if key_str.startswith("exam"):
        exam_info = EXAMS.get(key_str)
        if exam_info:
            print(f"✅ {exam_info['name']} 已完成！")
        else:
            print(f"✅ {key_str} 已标记完成！")
    else:
        n = int(key_str)
        print(f"✅ Lesson {n}: {LESSONS[n]} 已完成！")

    # 检查下一项
    all_items = []
    for items in STAGES.values():
        all_items.extend(items)

    current_idx = None
    for i, item in enumerate(all_items):
        if str(item) == key_str:
            current_idx = i
            break

    if current_idx is not None and current_idx + 1 < len(all_items):
        next_item = all_items[current_idx + 1]
        if isinstance(next_item, int):
            print(f"🔜 下一课: Lesson {next_item} - {LESSONS[next_item]}")
        else:
            exam_info = EXAMS.get(next_item, {})
            print(f"🔜 下一个: {exam_info.get('name', next_item)}")
    else:
        print("🎓 恭喜！后训练理论深化阶段全部完成！准备好进入阶段三了！")


def reset_lesson(progress, key_str):
    if key_str in progress["lessons"]:
        progress["lessons"][key_str] = {}
        print(f"🔄 {key_str} 已重置")
    else:
        print(f"⚠️  {key_str} 无进度记录，无需重置")


def main():
    progress = load_progress()

    if len(sys.argv) < 2 or sys.argv[1] == "show":
        show_progress(progress)
        return

    cmd = sys.argv[1]

    if cmd == "complete":
        if len(sys.argv) < 3:
            print("用法: python progress.py complete <N|examN> [score]")
            sys.exit(1)
        key_str = sys.argv[2].lower()
        # 处理分数（考试用）
        if len(sys.argv) >= 4 and key_str.startswith("exam"):
            try:
                score = int(sys.argv[3])
                if key_str not in progress["lessons"]:
                    progress["lessons"][key_str] = {}
                progress["lessons"][key_str]["score"] = score
                grade = get_grade(score)
                print(f"📊 成绩记录: {score}/100  {grade}")
            except ValueError:
                pass
        complete_lesson(progress, key_str)
        save_progress(progress)

    elif cmd == "reset":
        if len(sys.argv) < 3:
            print("用法: python progress.py reset <N|examN>")
            sys.exit(1)
        reset_lesson(progress, sys.argv[2].lower())
        save_progress(progress)

    elif cmd == "reset-all":
        confirm = input("⚠️  你确定要重置所有进度吗？(输入 yes 确认): ")
        if confirm.strip().lower() == "yes":
            progress = {"lessons": {}, "notes": [], "started_at": datetime.now().isoformat()}
            save_progress(progress)
            print("✅ 所有进度已重置")
        else:
            print("已取消")

    else:
        print(f"未知命令: {cmd}")
        print("可用命令: show | complete <N> | reset <N> | reset-all")


if __name__ == "__main__":
    main()
