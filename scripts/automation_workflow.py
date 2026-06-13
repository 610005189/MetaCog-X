# -*- coding: utf-8 -*-
"""自动化任务循环推进系统

功能：
1. 定期检查 specs 中的任务状态
2. 根据优先级自动推进待办任务
3. 执行验证并记录结果
4. 生成执行报告
"""
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TaskStatus:
    """任务状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AutomationConfig:
    """自动化配置"""
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.specs_dir = os.path.join(PROJECT_ROOT, ".trae", "specs")
        self.reports_dir = os.path.join(PROJECT_ROOT, "reports")
        self.check_interval = 60  # 秒
        self.max_iterations = 10
        self.auto_verify = True
        self.notify_on_complete = True
        
        # 确保目录存在
        os.makedirs(self.reports_dir, exist_ok=True)


class SpecAnalyzer:
    """Spec 分析器"""
    
    def __init__(self, specs_dir: str):
        self.specs_dir = specs_dir
        
    def find_active_specs(self) -> List[str]:
        """查找活跃的 spec 目录"""
        if not os.path.exists(self.specs_dir):
            return []
        
        active_specs = []
        for item in os.listdir(self.specs_dir):
            item_path = os.path.join(self.specs_dir, item)
            if os.path.isdir(item_path) and not item.startswith("_archived"):
                # 检查是否有 tasks.md
                tasks_file = os.path.join(item_path, "tasks.md")
                if os.path.exists(tasks_file):
                    active_specs.append(item)
        return active_specs
    
    def parse_tasks(self, spec_name: str) -> List[Dict]:
        """解析 tasks.md 中的任务"""
        tasks_file = os.path.join(self.specs_dir, spec_name, "tasks.md")
        if not os.path.exists(tasks_file):
            return []
        
        with open(tasks_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        tasks = []
        current_section = "unknown"
        
        for line in content.split("\n"):
            line = line.strip()
            
            # 检查章节
            if line.startswith("## "):
                current_section = line.replace("## ", "").strip()
                continue
                
            # 检查任务状态 - 更严格的匹配
            if line.startswith("- [x]"):
                status = TaskStatus.COMPLETED
            elif line.startswith("- [/]"):
                status = TaskStatus.IN_PROGRESS
            elif line.startswith("- [ ]"):
                status = TaskStatus.PENDING
            else:
                continue
                
            # 提取任务名称 - 跳过子任务
            if line.startswith("- [x] Task") or line.startswith("- [ ] Task") or line.startswith("- [/] Task"):
                # 找到任务行
                parts = line.split("]", 1)
                if len(parts) > 1:
                    task_name = parts[1].strip()
                    if task_name.startswith("Task"):
                        task_name = task_name.split(":", 1)[-1].strip()
                        
                    tasks.append({
                        "name": task_name,
                        "status": status,
                        "section": current_section,
                        "spec": spec_name
                    })
        
        return tasks
    
    def get_pending_tasks(self, spec_name: str) -> List[Dict]:
        """获取待办任务"""
        all_tasks = self.parse_tasks(spec_name)
        return [t for t in all_tasks if t["status"] == TaskStatus.PENDING]
    
    def get_task_priority(self, task: Dict) -> int:
        """估算任务优先级"""
        name = task["name"].lower()
        section = task["section"].lower()
        
        # 基于名称
        if "p0" in name or "critical" in name or "核心" in name:
            return 0
        elif "p1" in name or "important" in name or "重要" in name:
            return 1
        elif "p2" in name or "nice" in name or "优化" in name:
            return 2
        
        # 基于章节
        if "阶段1" in section or "phase 1" in section:
            return 0
        elif "阶段2" in section or "phase 2" in section:
            return 1
        elif "阶段3" in section or "phase 3" in section:
            return 1
        elif "阶段4" in section or "phase 4" in section:
            return 2
        
        return 1


class TaskExecutor:
    """任务执行器"""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        
    def find_task_script(self, task: Dict) -> Optional[str]:
        """查找任务对应的脚本"""
        name = task["name"].lower()
        
        # 基于任务名称映射到脚本
        mappings = {
            "training": "runs/run_medium_train.py",
            "训练": "runs/run_medium_train.py",
            "imitation": "runs/verify_imitation_learning.py",
            "模仿学习": "runs/verify_imitation_learning.py",
            "ablation": "runs/verify_ablation.py",
            "消融": "runs/verify_ablation.py",
            "intervention": "training/intervention_training.py",
            "干预": "training/intervention_training.py",
        }
        
        for key, script in mappings.items():
            if key in name:
                script_path = os.path.join(self.project_root, script)
                if os.path.exists(script_path):
                    return script_path
        
        return None
    
    def execute_task(self, task: Dict) -> Tuple[bool, str]:
        """执行任务"""
        print(f"\n执行任务: {task['name']}")
        print(f"Spec: {task['spec']}")
        
        # 查找脚本
        script = self.find_task_script(task)
        if script:
            print(f"找到脚本: {script}")
            try:
                import subprocess
                result = subprocess.run(
                    [sys.executable, script],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    return True, result.stdout
                else:
                    return False, result.stderr
            except Exception as e:
                return False, str(e)
        else:
            # 没有对应脚本，标记为需要手动处理
            return False, "No automated script found for this task"
    
    def verify_task(self, task: Dict) -> bool:
        """验证任务"""
        # 简化的验证：检查相关文件是否存在
        name = task["name"].lower()
        
        if "training" in name or "训练" in name:
            return os.path.exists(os.path.join(self.project_root, "runs", "run_medium_train.py"))
        elif "imitation" in name or "模仿" in name:
            return os.path.exists(os.path.join(self.project_root, "runs", "verify_imitation_learning.py"))
        elif "ablation" in name or "消融" in name:
            return os.path.exists(os.path.join(self.project_root, "runs", "verify_ablation.py"))
        elif "intervention" in name or "干预" in name:
            return os.path.exists(os.path.join(self.project_root, "training", "intervention_training.py"))
        
        return True  # 默认通过


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, reports_dir: str):
        self.reports_dir = reports_dir
        
    def generate_report(self, execution_results: List[Dict]) -> str:
        """生成执行报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.reports_dir, f"automation_report_{timestamp}.md")
        
        total = len(execution_results)
        succeeded = sum(1 for r in execution_results if r["success"])
        failed = total - succeeded
        
        content = f"""# 自动化任务执行报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**执行任务数**: {total}
**成功**: {succeeded}
**失败**: {failed}

---

## 执行详情

| 任务 | Spec | 状态 | 消息 |
|------|------|------|------|
"""
        
        for result in execution_results:
            status_icon = "✓" if result["success"] else "✗"
            message = result["message"][:50] + "..." if len(result["message"]) > 50 else result["message"]
            content += f"| {result['task_name']} | {result['spec']} | {status_icon} | {message} |\n"
        
        content += f"""

---

## 总结

执行时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

"""
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(content)
            
        return report_file


class AutomationWorkflow:
    """自动化工作流"""
    
    def __init__(self, config: Optional[AutomationConfig] = None):
        self.config = config or AutomationConfig()
        self.analyzer = SpecAnalyzer(self.config.specs_dir)
        self.executor = TaskExecutor(self.config.project_root)
        self.report_generator = ReportGenerator(self.config.reports_dir)
        
    def run(self, max_iterations: Optional[int] = None) -> List[Dict]:
        """运行自动化工作流"""
        max_iter = max_iterations or self.config.max_iterations
        
        print("=" * 60)
        print("MetaCog-X 自动化任务执行")
        print("=" * 60)
        print(f"Spec 目录: {self.config.specs_dir}")
        print(f"最大迭代: {max_iter}")
        
        # 查找活跃 specs
        active_specs = self.analyzer.find_active_specs()
        print(f"\n发现 {len(active_specs)} 个活跃 specs:")
        for spec in active_specs:
            print(f"  - {spec}")
        
        execution_results = []
        
        for iteration in range(max_iter):
            print(f"\n--- 迭代 {iteration + 1}/{max_iter} ---")
            
            # 收集所有待办任务
            all_pending = []
            for spec in active_specs:
                pending = self.analyzer.get_pending_tasks(spec)
                for task in pending:
                    task["priority"] = self.analyzer.get_task_priority(task)
                all_pending.extend(pending)
            
            if not all_pending:
                print("没有待办任务，工作流完成!")
                break
                
            # 按优先级排序
            all_pending.sort(key=lambda t: t["priority"])
            
            # 执行最高优先级的任务
            task = all_pending[0]
            print(f"\n选择任务: {task['name']} (优先级: {task['priority']})")
            
            success, message = self.executor.execute_task(task)
            
            result = {
                "task_name": task["name"],
                "spec": task["spec"],
                "success": success,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            execution_results.append(result)
            
            if success:
                print(f"✓ 任务成功")
            else:
                print(f"✗ 任务失败: {message}")
        
        # 生成报告
        if execution_results:
            report_file = self.report_generator.generate_report(execution_results)
            print(f"\n报告已生成: {report_file}")
        
        return execution_results


def main():
    """主函数"""
    print("MetaCog-X 自动化任务循环推进系统")
    print("=" * 60)
    
    # 创建配置
    config = AutomationConfig()
    
    # 运行工作流
    workflow = AutomationWorkflow(config)
    results = workflow.run(max_iterations=3)  # 默认最多执行3个任务
    
    # 打印总结
    print("\n" + "=" * 60)
    print("执行总结")
    print("=" * 60)
    total = len(results)
    succeeded = sum(1 for r in results if r["success"])
    print(f"总任务数: {total}")
    print(f"成功: {succeeded}")
    print(f"失败: {total - succeeded}")


if __name__ == "__main__":
    main()