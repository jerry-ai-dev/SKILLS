"""
阶段三：开源项目研读 - 学习进度管理脚本
用法：
  python progress.py show                       # 查看当前进度
  python progress.py complete <N>               # 标记第 N 课完成（N=0-10，0 为导论课）
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
    0:  "阶段三导论：定位、目标、方法、节奏",
    1:  "TRL 库全景 & SFTTrainer 源码",
    2:  "TRL GRPOTrainer 源码精读",
    3:  "TRL 数据流水线 & Reward 设计",
    4:  "Open-R1 项目架构总览",
    5:  "Open-R1 SFT 训练流程",
    6:  "Open-R1 GRPO 训练流程",
    7:  "Open-R1 奖励函数 & 评估体系",
    8:  "SimpleRL-Zoo 小模型 RL 实验",
    9:  "通用模式提炼 & 代码模板",
    10: "实战规划：你的 SFT+GRPO Pipeline",
}

EXAMS = {
    "exam1": {"name": "📝 阶段考试 1: TRL 库", "after": 3, "covers": "Lesson 1-3"},
    "exam2": {"name": "📝 阶段考试 2: Open-R1", "after": 7, "covers": "Lesson 4-7"},
    "exam3": {"name": "🎓 期末综合考试", "after": 10, "covers": "Lesson 1-10"},
}

STAGES = {
    "导论":                          [0],
    "第一阶段 TRL 库精读":       [1, 2, 3, "exam1"],
    "第二阶段 Open-R1 深度拆解":  [4, 5, 6, 7, "exam2"],
    "第三阶段 整合与实战规划":     [8, 9, 10, "exam3"],
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
    print(f"\n📊 开源项目研读进度: [{bar}] {completed}/{total}")
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


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    progress = load_progress()
    cmd = sys.argv[1]

    if cmd == "show":
        show_progress(progress)

    elif cmd == "complete":
        if len(sys.argv) < 3:
            print("用法: python progress.py complete <N> 或 complete exam<N> [score]")
            return
        target = sys.argv[2]

        if target.startswith("exam"):
            key = target
            if key not in EXAMS:
                print(f"❌ 无效的考试编号: {key}，可选: {list(EXAMS.keys())}")
                return
            score = int(sys.argv[3]) if len(sys.argv) > 3 else None
            complete_lesson(progress, key)
            if score is not None:
                progress["lessons"][key]["score"] = score
                grade = get_grade(score)
                print(f"📊 成绩记录: {score}/100  {grade}")
            print(f"✅ {EXAMS[key]['name']} 已完成！")
            # 检查是否全部完成
            all_done = all(
                progress["lessons"].get(str(item), {}).get("completed", False)
                for items in STAGES.values() for item in items
            )
            if all_done:
                print("🎓 恭喜！开源项目研读阶段全部完成！准备好进入阶段四了！")
        else:
            try:
                n = int(target)
            except ValueError:
                print(f"❌ 无效参数: {target}")
                return
            if n not in LESSONS:
                valid = sorted(LESSONS.keys())
                print(f"❌ 无效的课程编号: {n}，可选: {valid}")
                return
            complete_lesson(progress, str(n))
            print(f"✅ Lesson {n}: {LESSONS[n]} 已完成！")

        save_progress(progress)

    elif cmd == "reset":
        if len(sys.argv) < 3:
            print("用法: python progress.py reset <N>")
            return
        target = sys.argv[2]
        key = target if target.startswith("exam") else target
        if key in progress["lessons"]:
            del progress["lessons"][key]
            print(f"🔄 已重置: {key}")
            save_progress(progress)
        else:
            print(f"⚠️ {key} 尚未有记录")

    elif cmd == "reset-all":
        progress["lessons"] = {}
        progress["started_at"] = datetime.now().isoformat()
        save_progress(progress)
        print("🔄 所有进度已重置")

    else:
        print(f"❌ 未知命令: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
