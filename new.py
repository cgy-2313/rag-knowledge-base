from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil
import os

embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key="sk-htuzymdmwbisppkrndpwyyjdnvxmqhebkavhbfurjjacjnby",
    base_url="https://api.siliconflow.cn/v1"
)

# 清空数据库
if os.path.exists("./rag_db"):
    shutil.rmtree("./rag_db")

# 加载文档
loader = TextLoader("knowledge_base.txt", encoding="utf-8")
docs = loader.load()
print(f"文档字符数：{len(docs[0].page_content)}")
print(f"文档前50字：{docs[0].page_content[:50]}")

# 切片
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(docs)
print(f"切片数量：{len(chunks)}")

# 存入数据库
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./rag_db"
)
print(f"向量数量：{vectorstore._collection.count()}")

# 把所有7个片段都打印出来
print("数据库里所有片段：")
all_results = vectorstore.similarity_search("住宿", k=7)
for i, r in enumerate(all_results):
    print(f"\n片段{i+1}：")
    print(r.page_content)
    print(f"相似度元数据：{r.metadata}")

# 测试检索
# results = vectorstore.similarity_search("广州住宿限额", k=5)
# print(f"\n检索结果：")
# for i, r in enumerate(results):
#     print(f"片段{i+1}：{r.page_content}")

    