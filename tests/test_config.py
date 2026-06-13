"""配置测试"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MetaCogXConfig


def test_config_default():
    """测试默认配置"""
    config = MetaCogXConfig()
    
    assert config.d_model == 512, f"默认d_model不正确: {config.d_model}"
    assert config.num_layers == 12, f"默认num_layers不正确: {config.num_layers}"
    assert config.num_heads == 8, f"默认num_heads不正确: {config.num_heads}"
    assert config.d_head == 64, f"自动计算d_head不正确: {config.d_head}"
    
    print("✓ 默认配置测试通过")


def test_config_tiny():
    """测试tiny配置 (d_model=128)"""
    config = MetaCogXConfig.tiny()
    
    assert config.d_model == 128, f"tiny d_model不正确: {config.d_model}"
    assert config.d_meta == 16, f"tiny d_meta不正确: {config.d_meta}"
    assert config.d_aware == 8, f"tiny d_aware不正确: {config.d_aware}"
    assert config.num_layers == 4, f"tiny num_layers不正确: {config.num_layers}"
    assert config.num_heads == 4, f"tiny num_heads不正确: {config.num_heads}"
    assert config.d_ffn == 512, f"tiny d_ffn不正确: {config.d_ffn}"
    assert config.d_head == 32, f"tiny d_head不正确: {config.d_head}"
    
    print("✓ tiny配置测试通过")


def test_config_small():
    """测试small配置 (d_model=256)"""
    config = MetaCogXConfig.small()
    
    assert config.d_model == 256, f"small d_model不正确: {config.d_model}"
    assert config.d_meta == 24, f"small d_meta不正确: {config.d_meta}"
    assert config.d_aware == 12, f"small d_aware不正确: {config.d_aware}"
    assert config.num_layers == 8, f"small num_layers不正确: {config.num_layers}"
    assert config.num_heads == 8, f"small num_heads不正确: {config.num_heads}"
    assert config.d_ffn == 1024, f"small d_ffn不正确: {config.d_ffn}"
    assert config.d_head == 32, f"small d_head不正确: {config.d_head}"
    
    print("✓ small配置测试通过")


def test_config_medium():
    """测试medium配置 (d_model=512)"""
    config = MetaCogXConfig.medium()
    
    assert config.d_model == 512, f"medium d_model不正确: {config.d_model}"
    assert config.num_layers == 12, f"medium num_layers不正确: {config.num_layers}"
    assert config.num_heads == 8, f"medium num_heads不正确: {config.num_heads}"
    assert config.d_head == 64, f"medium d_head不正确: {config.d_head}"
    
    print("✓ medium配置测试通过")


def test_config_large():
    """测试large配置 (d_model=1024)"""
    config = MetaCogXConfig.large()
    
    assert config.d_model == 1024, f"large d_model不正确: {config.d_model}"
    assert config.num_layers == 16, f"large num_layers不正确: {config.num_layers}"
    assert config.num_heads == 16, f"large num_heads不正确: {config.num_heads}"
    assert config.d_head == 64, f"large d_head不正确: {config.d_head}"
    
    print("✓ large配置测试通过")


def test_config_str():
    """测试配置字符串输出"""
    config = MetaCogXConfig.small()
    str_repr = str(config)
    
    assert "d_model=256" in str_repr, "配置字符串不包含d_model"
    assert "num_layers=8" in str_repr, "配置字符串不包含num_layers"
    
    print("✓ 配置字符串测试通过")


if __name__ == "__main__":
    print("=" * 50)
    print("配置测试")
    print("=" * 50)
    
    test_config_default()
    test_config_tiny()
    test_config_small()
    test_config_medium()
    test_config_large()
    test_config_str()
    
    print("=" * 50)
    print("所有测试通过！")
    print("=" * 50)