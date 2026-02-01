from pathlib import Path

def find_project_root_with_tests(start_path=None):
    """
    从 start_path 开始向上查找，直到找到包含 'tests' 目录的父目录。
    返回该父目录的 Path 对象。
    如果没找到，抛出 FileNotFoundError。
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path).resolve()

    # 限制向上查找，避免无限循环（比如到根目录为止）
    for parent in [start_path] + list(start_path.parents):
        if (parent / "tests").is_dir():
            return parent

    raise FileNotFoundError("未找到包含 'tests' 目录的项目根目录")
