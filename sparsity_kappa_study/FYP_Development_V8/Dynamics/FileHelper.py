import os
import json
import shutil

def Clear_Folder(folder_path='Output'):
    """
    删除指定文件夹下的所有文件。

    参数：
        folder_path (str): 文件夹的路径。
    """
    if not os.path.exists(folder_path):
        print(f"路径 {folder_path} 不存在。")
        return

    if not os.path.isdir(folder_path):
        print(f"路径 {folder_path} 不是文件夹。")
        return

    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # 检查是否是文件
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                # 如果需要删除子文件夹，可以使用 shutil.rmtree
                shutil.rmtree(file_path)
        print(f"已删除文件夹 {folder_path} 下的所有文件。")
    except Exception as e:
        print(f"删除文件时出错: {e}")

def Save_JSON(obj, file):
    """将对象储存到JSON文件"""
    with open(file, 'w') as f:
        json.dump(vars(obj), f, indent=4)

def Read_JSON(file):
    """从JSON文件中读取数据"""
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data