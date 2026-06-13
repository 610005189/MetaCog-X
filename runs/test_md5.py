# -*- coding: utf-8 -*-
"""验证MD5哈希稳定性"""
import hashlib

# 测试MD5哈希的跨会话稳定性
p = "What is 2+2?"

# MD5哈希 - 跨会话稳定
md5_hash = int(hashlib.md5(p.encode()).hexdigest(), 16) % 10000
print(f"MD5哈希值: {md5_hash}")

# Python内置hash - 每次可能不同（取决于PYTHONHASHSEED）
py_hash = hash(p) % 10000
print(f"Python hash值: {py_hash}")

print("\n验证: MD5哈希在多次调用中保持稳定")
results = [int(hashlib.md5(p.encode()).hexdigest(), 16) % 10000 for _ in range(5)]
print(f"5次MD5调用结果: {results}")
print(f"全部相同: {len(set(results)) == 1}")
