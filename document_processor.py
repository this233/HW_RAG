from typing import List, Optional
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import os

class DocumentProcessor:
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 separators: List[str] = ["\n\n", "\n", "。", "！", "？", ".", "!", "?"]):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        # 定义 Markdown 标题层级
        self.headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
            ("#####", "h5"),
            ("######", "h6"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )
        
        # 初始化语义分块器
        self.semantic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=True,
            is_separator_regex=False
        )

    def _semantic_split(self, markdown_split: dict) -> List[dict]:
        """
        对 Markdown 分块后的内容进行进一步的语义分块
        
        Args:
            markdown_split: Markdown 分块结果
            
        Returns:
            List[dict]: 语义分块结果列表
        """
        # 获取文本内容和元数据
        text = markdown_split.page_content
        metadata = markdown_split.metadata.copy()
        
        # 进行语义分块
        semantic_splits = self.semantic_splitter.split_text(text)
        
        # 为每个语义分块创建带有完整元数据的字典
        result = []
        for split in semantic_splits:
            # 创建新的分块对象，保持与原始分块相同的接口
            split_dict = type('Split', (), {
                'page_content': split,
                'metadata': metadata.copy()
            })()
            result.append(split_dict)
            
        return result

    def process_file(self, file_path: str) -> List[dict]:
        """
        处理单个文件并返回分块结果
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[dict]: 包含分块内容和元数据的列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # 只处理 Markdown 文件
        if not file_path.lower().endswith('.md'):
            raise ValueError(f"Only markdown files are supported, got: {file_path}")
            
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 使用 MarkdownHeaderTextSplitter 进行分块
        markdown_splits = self.markdown_splitter.split_text(content)
        
        # 对每个 Markdown 分块进行语义分块
        all_splits = []
        for split in markdown_splits:
            # 如果块太大，才要再进行精细分块
            # 检查块的大小是否超过设定的chunk_size
            split.metadata['source'] = file_path
            if len(split.page_content) > self.chunk_size:
                semantic_splits = self._semantic_split(split)
            else:
                # 如果块大小合适，直接使用原块
                semantic_splits = [split]

            # 添加文件路径信息
            
            all_splits.extend(semantic_splits)
            
        return all_splits

    def process_directory(self, directory_path: str) -> List[dict]:
        """
        处理目录下的所有 Markdown 文件
        
        Args:
            directory_path: 目录路径
            
        Returns:
            List[dict]: 所有文件的分块结果合并列表
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        all_splits = []
        
        # 遍历目录下的所有文件
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.md'):
                    file_path = os.path.join(root, file)
                    try:
                        splits = self.process_file(file_path)
                        all_splits.extend(splits)
                    except Exception as e:
                        print(f"Error processing file {file_path}: {str(e)}")
                        continue
                        
        return all_splits

# 使用示例
if __name__ == "__main__":
    # 创建处理器实例，可以自定义分块参数
    processor = DocumentProcessor(
        chunk_size=500,  # 每个分块的最大字符数
        chunk_overlap=50,  # 相邻分块的重叠字符数
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"]  # 分隔符列表
    )
    
    # 处理单个文件
    splits = processor.process_file("docs/zh-cn/release-notes/api-diff/Beta5 to v3.2-Release/js-apidiff-ability.md")
    for split in splits:
        print(split.page_content)
        print(split.metadata)
        print("-"*100)
    
    # 处理整个目录
    # splits = processor.process_directory("docs/zh-cn/")
    
    # 打印分块结果示例
    # for split in splits:
    #     print("Content:", split.page_content[:100])  # 只打印前100个字符
    #     print("Metadata:", split.metadata)
    #     print("-" * 80)
