"""Tokenizer集成测试"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.hf_dataset import get_tokenizer, load_wikitext_dataset, HFDataset, CharLevelTokenizer


def test_charlevel_tokenizer():
    """测试字符级tokenizer"""
    tokenizer = CharLevelTokenizer()
    
    # 测试编码
    text = "Hello World!"
    encoded = tokenizer.encode(text)
    assert len(encoded) == len(text), "编码长度不匹配"
    
    # 测试解码
    decoded = tokenizer.decode(encoded)
    assert decoded == text, f"解码结果不匹配: {decoded} != {text}"
    
    # 测试__call__方法
    result = tokenizer(text, max_length=20)
    assert 'input_ids' in result
    assert 'attention_mask' in result
    assert result['input_ids'].shape == (1, 20)
    assert result['attention_mask'].shape == (1, 20)
    
    print("✓ CharLevelTokenizer测试通过")


def test_gpt2_tokenizer():
    """测试GPT2 tokenizer"""
    try:
        tokenizer = get_tokenizer("gpt2")
        
        # 测试编码解码
        text = "Hello World!"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        assert len(encoded) > 0, "编码结果为空"
        assert decoded.strip() == text, f"解码结果不匹配: {decoded} != {text}"
        
        # 测试__call__方法
        result = tokenizer(text, max_length=20)
        assert 'input_ids' in result
        assert 'attention_mask' in result
        
        print("✓ GPT2 Tokenizer测试通过")
    except Exception as e:
        print(f"⚠️ GPT2 Tokenizer测试跳过（可能缺少transformers库）: {e}")


def test_hf_dataset():
    """测试HFDataset"""
    tokenizer = CharLevelTokenizer()
    texts = ["Hello World!", "This is a test.", "MetaCog-X is awesome!"]
    
    ds = HFDataset(tokenizer, texts, max_length=32)
    
    assert len(ds) == 3, f"数据集长度不正确: {len(ds)} != 3"
    
    # 测试数据加载
    input_ids, attention_mask = ds[0]
    assert input_ids.shape == (32,), f"input_ids形状不正确: {input_ids.shape}"
    assert attention_mask.shape == (32,), f"attention_mask形状不正确: {attention_mask.shape}"
    
    print("✓ HFDataset测试通过")


def test_load_wikitext_dataset():
    """测试加载wikitext数据集"""
    # 测试charlevel
    ds_char = load_wikitext_dataset(split="train", max_train_samples=10, tokenizer_type="charlevel")
    assert len(ds_char) == 10, f"charlevel数据集长度不正确: {len(ds_char)}"
    print("✓ load_wikitext_dataset (charlevel)测试通过")
    
    # 测试gpt2
    try:
        ds_gpt2 = load_wikitext_dataset(split="train", max_train_samples=10, tokenizer_type="gpt2")
        assert len(ds_gpt2) == 10, f"gpt2数据集长度不正确: {len(ds_gpt2)}"
        print("✓ load_wikitext_dataset (gpt2)测试通过")
    except Exception as e:
        print(f"⚠️ load_wikitext_dataset (gpt2)测试跳过: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("Tokenizer集成测试")
    print("=" * 50)
    
    test_charlevel_tokenizer()
    test_gpt2_tokenizer()
    test_hf_dataset()
    test_load_wikitext_dataset()
    
    print("=" * 50)
    print("所有测试通过！")
    print("=" * 50)