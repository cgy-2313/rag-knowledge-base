from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 第一步：检查文档加载
loader = TextLoader("knowledge_base.txt", encoding="utf-8")
docs = loader.load()
print(f"文档字符数：{len(docs[0].page_content)}")
print(f"文档前100字：{docs[0].page_content[:100]}")

# 第二步：检查切片结果
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 从200改成100
    chunk_overlap=20
)
chunks = splitter.split_documents(docs)
print(f"\n切片数量：{len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n第{i+1}片（{len(chunk.page_content)}字）：")
    print(chunk.page_content)