#!/usr/bin/env python3
"""
快速查看 info.npy 文件的结构和内容
"""
import numpy as np
import os

def inspect_npy_file(file_path):
    """查看 .npy 文件的详细信息"""
    print("=" * 60)
    print(f"正在查看文件: {file_path}")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return
    
    # 加载文件
    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"错误: 无法加载文件: {e}")
        return
    
    # 如果是字典（item()）
    if isinstance(data, np.ndarray) and data.dtype == object:
        try:
            data = data.item()
        except:
            pass
    
    print(f"\n数据类型: {type(data)}")
    
    # 如果是字典
    if isinstance(data, dict):
        print(f"\n字典包含 {len(data)} 个键:")
        print("-" * 60)
        
        for key, value in data.items():
            print(f"\n键名: '{key}'")
            print(f"  类型: {type(value)}")
            
            if isinstance(value, (list, np.ndarray)):
                if isinstance(value, list):
                    print(f"  列表长度: {len(value)}")
                    if len(value) > 0:
                        print(f"  第一个元素类型: {type(value[0])}")
                        if isinstance(value[0], np.ndarray):
                            print(f"  第一个元素形状: {value[0].shape}")
                            print(f"  第一个元素数据类型: {value[0].dtype}")
                            print(f"  第一个元素内容:\n    {value[0]}")
                        else:
                            print(f"  第一个元素内容: {value[0]}")
                elif isinstance(value, np.ndarray):
                    print(f"  数组形状: {value.shape}")
                    print(f"  数据类型: {value.dtype}")
                    print(f"  数组内容:\n{value}")
            else:
                print(f"  值: {value}")
            
            print()
    
    # 如果是 numpy 数组
    elif isinstance(data, np.ndarray):
        print(f"\n数组形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"数组内容:\n{data}")
    
    # 其他类型
    else:
        print(f"\n内容: {data}")
    
    print("=" * 60)

if __name__ == "__main__":
    # 默认路径
    default_path = "traj_demo/imgs/info.npy"
    
    # 如果提供了命令行参数，使用参数作为路径
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_path
    
    inspect_npy_file(file_path)

