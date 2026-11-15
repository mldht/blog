from modelscope import MsDataset
import pandas as pd
import os

# 方案一：尝试用 MsDataset 加载（类似 datasets 的封装）
try:
    # 注意：split 参数可能需要根据 ModelScope 上的实际结构调整
    ds = MsDataset.load('gaia-benchmark/GAIA', subset_name='2023_level1', split='validation')
    print(f"Number of Level 1 problems: {len(ds)}")
except Exception as e:
    print(f"MsDataset 加载失败: {e}")
