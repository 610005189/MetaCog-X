# 贡献指南

感谢您对 MetaCog-X 项目的关注！本文档将帮助您了解如何为项目做出贡献。

## 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [开发环境设置](#开发环境设置)
- [代码风格](#代码风格)
- [提交规范](#提交规范)
- [Pull Request 流程](#pull-request-流程)
- [问题报告](#问题报告)

## 行为准则

请保持友善和尊重。我们致力于为所有人提供友好、安全和受欢迎的环境。

## 如何贡献

### 报告 Bug

如果您发现了 bug，请通过 [GitHub Issues](https://github.com/610005189/MetaCog-X/issues) 提交报告。提交时请包含：

- 清晰的标题和描述
- 复现步骤
- 预期行为和实际行为
- 环境信息（Python 版本、PyTorch 版本、操作系统）
- 相关日志或截图

### 提出新功能

欢迎提出新功能建议！请在 Issue 中详细描述：

- 功能的用途和价值
- 可能的实现方式
- 是否愿意自己实现

### 提交代码

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 开发环境设置

### 系统要求

- Python 3.8+
- PyTorch 1.12+ (推荐 2.0+)

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/610005189/MetaCog-X.git
cd MetaCog-X

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install torch numpy scikit-learn transformers

# 安装开发依赖
pip install pytest black flake8
```

### 运行测试

```bash
# 运行所有测试
python tests/run_tests.py

# 运行特定测试
python -m pytest tests/test_unit.py -v
```

## 代码风格

### Python 代码规范

- 遵循 [PEP 8](https://peps.python.org/pep-0008/) 风格指南
- 使用 4 空格缩进
- 最大行长度 100 字符
- 使用有意义的变量和函数名称

### 代码格式化

我们使用 Black 进行代码格式化：

```bash
# 格式化代码
black .

# 检查格式
black --check .
```

### 类型注解

推荐使用类型注解提高代码可读性：

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...
```

### 文档字符串

使用 Google 风格的文档字符串：

```python
def compute_dilemma_score(self, attention_entropy: float, logits_std: float) -> float:
    """计算困境分数。

    Args:
        attention_entropy: 注意力熵值
        logits_std: logits 标准差

    Returns:
        困境分数，范围 [0, 1]
    """
    ...
```

## 提交规范

我们使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

### 提交格式

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### 提交类型

| 类型 | 描述 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档更新 |
| `style` | 代码格式（不影响功能） |
| `refactor` | 代码重构 |
| `perf` | 性能优化 |
| `test` | 测试相关 |
| `chore` | 构建/工具相关 |

### 示例

```
feat(models): add sparse meta controller

- Implement sparse attention mechanism
- Add configurable sparsity threshold
- Include unit tests

Closes #123
```

## Pull Request 流程

### 提交前检查

- [ ] 代码通过所有测试
- [ ] 代码符合风格规范
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 提交信息符合规范

### 审核流程

1. 创建 PR 后，维护者会进行代码审核
2. 根据反馈进行必要的修改
3. 通过审核后，PR 将被合并

### PR 标题格式

使用与提交信息相同的格式：

```
feat(models): add new attention mechanism
fix(training): resolve gradient explosion issue
docs: update installation guide
```

## 问题报告

如果您在使用过程中遇到问题，可以通过以下方式获取帮助：

1. 查看 [README.md](README.md) 中的文档
2. 搜索 [Issues](https://github.com/610005189/MetaCog-X/issues) 是否有类似问题
3. 创建新的 Issue，详细描述问题

## 项目结构

了解项目结构有助于您更好地贡献代码：

```
MetaCog-X/
├── models/           # 核心模型架构
├── training/         # 训练框架
├── runs/             # 实验入口脚本
├── data/             # 数据处理工具
├── scripts/          # 分析工具
├── tests/            # 测试文件
└── config.py         # 配置文件
```

## 联系方式

- GitHub Issues: https://github.com/610005189/MetaCog-X/issues
- 项目主页: https://github.com/610005189/MetaCog-X

---

再次感谢您对 MetaCog-X 的贡献！