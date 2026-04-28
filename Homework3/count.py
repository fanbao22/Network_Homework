import json
import os
import re


class QiyanJuejuLoader:
    def __init__(self, file_paths):
        """
        初始化数据加载对象
        :param file_paths: list, 包含需要解析的JSON文件路径列表
        """
        self.file_paths = file_paths

    def _is_qiyan_jueju(self, poem):
        """
        内部方法：判断一首诗是否为七言绝句
        :param poem: dict, 单首诗的字典数据
        :return: bool, True表示是七言绝句，False表示不是
        """
        paragraphs = poem.get("paragraphs", [])
        if not paragraphs:
            return False

        # 将所有段落合并为一个完整的字符串
        content = "".join(paragraphs)

        # 使用正则表达式按照常见中文标点符号进行切分，并过滤掉切分产生的空字符串
        # 常见标点：逗号、句号、感叹号、问号
        sentences = [s for s in re.split(r'[，。！？]', content) if s.strip()]

        # 七言绝句的硬性条件一：必须恰好是4句
        if len(sentences) != 4:
            return False

        # 七言绝句的硬性条件二：每一句必须恰好是7个字
        for sentence in sentences:
            if len(sentence) != 7:
                return False

        return True

    def load_all(self):
        """
        一次性加载：从所有配置的文件中加载并返回所有七言绝句
        注意：如果文件极大，这种方式可能会占用较多内存
        :return: list, 包含所有七绝的列表
        """
        results = []
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                print(f"警告：文件 {file_path} 不存在，已跳过。")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for poem in data:
                        if self._is_qiyan_jueju(poem):
                            results.append(poem)
            except Exception as e:
                print(f"读取或解析 {file_path} 时发生错误: {e}")

        return results

    def load_generator(self):
        """
        生成器加载：以流的形式逐个返回七言绝句
        推荐使用此方法，有效节省内存，特别适合处理成千上万首诗的合集文件
        :yield: dict, 单首七绝的数据
        """
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                print(f"警告：文件 {file_path} 不存在，已跳过。")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for poem in data:
                        if self._is_qiyan_jueju(poem):
                            yield poem
            except Exception as e:
                print(f"读取或解析 {file_path} 时发生错误: {e}")


# ==========================================
# 示例用法
# ==========================================
if __name__ == "__main__":
    # 使用你上传的JSON文件列表
    json_files = [
        "poet.song.40000.json",
        "poet.song.41000.json",
        "poet.song.42000.json",
        "poet.song.43000.json"
    ]

    # 实例化数据加载器
    loader = QiyanJuejuLoader(json_files)

    # 使用生成器遍历
    print("开始使用生成器检索七言绝句...")
    count = 0
    for poem in loader.load_generator():
        # 仅打印前两首作为演示
        if count < 2:
            print(f"\n标题: {poem.get('title', '无题')}")
            print(f"作者: {poem.get('author', '佚名')}")
            print(f"内容:\n" + "\n".join(poem.get('paragraphs', [])))
        count += 1

    print(f"\n检索完成，共找到 {count} 首七言绝句。")