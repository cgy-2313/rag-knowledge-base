from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="sk-b99631e460b0473da7bcce584675814d",
    base_url="https://api.deepseek.com",
    temperature=0
)

# ============================================================
# 示例1：加载文本文件
# ============================================================

print("=" * 50)
print("示例1：加载文本文件")
print("=" * 50)

# 先创建一个测试用的txt文件
with open("test.txt", "w", encoding="utf-8") as f:
    f.write("""
大模型应用开发指南

第一章：什么是大模型
大模型是指参数量巨大的神经网络模型，通常在海量数据上训练。
代表性的大模型包括GPT-4、Claude、DeepSeek等。
这些模型能够理解和生成自然语言，完成各种复杂任务。

第二章：RAG技术
RAG全称检索增强生成，是目前最主流的大模型应用开发方式。
它的核心思路是：不把所有文档塞进上下文，而是只检索最相关的片段。
RAG系统由三个核心部分组成：文档切片、向量检索、生成回答。

第三章：向量数据库
向量数据库用来存储文本的向量表示。
常用的向量数据库有Chroma、Faiss、Pinecone等。
Chroma是最适合本地开发的向量数据库，简单易用。
""")

# 加载文件
loader = TextLoader("test.txt", encoding="utf-8")
docs = loader.load()

print(f"加载了{len(docs)}个文档")
print(f"文档内容前100字：{docs[0].page_content[:100]}")
print(f"文档元数据：{docs[0].metadata}")


# ============================================================
# 示例2：文档切片（重点）
# ============================================================

print("\n" + "=" * 50)
print("示例2：文档切片")
print("=" * 50)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,     # 每块最多100个字符
    chunk_overlap=20,   # 相邻块之间重叠20个字符
)

chunks = splitter.split_documents(docs)

print(f"切片前：1个文档")
print(f"切片后：{len(chunks)}个块")
print()

for i, chunk in enumerate(chunks):
    print(f"第{i+1}块（{len(chunk.page_content)}字）：")
    print(chunk.page_content)
    print()


# ============================================================
# 示例3：理解chunk_overlap的作用
# ============================================================

print("=" * 50)
print("示例3：理解chunk_overlap")
print("=" * 50)

# 没有overlap
splitter_no_overlap = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

# 有overlap
splitter_with_overlap = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=30
)

chunks_no = splitter_no_overlap.split_documents(docs)
chunks_with = splitter_with_overlap.split_documents(docs)

print(f"没有overlap：{len(chunks_no)}块")
print(f"有overlap：{len(chunks_with)}块")
print()
print("没有overlap的第1块结尾：")
print(chunks_no[0].page_content[-30:])
print("有overlap的第2块开头：")
print(chunks_with[1].page_content[:30])
print("（可以看到有overlap时，第2块开头和第1块结尾有重叠）")


# ============================================================
# 示例4：向量化和存入Chroma（重点）
# ============================================================

print("\n" + "=" * 50)
print("示例4：向量化存入Chroma")
print("=" * 50)

# 用DeepSeek兼容的embedding
# 注意：这里用硅基流动的embedding接口，DeepSeek没有提供embedding
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key="sk-htuzymdmwbisppkrndpwyyjdnvxmqhebkavhbfurjjacjnby",
    base_url="https://api.siliconflow.cn/v1"
)

# 把切好的块存入Chroma向量数据库
if os.path.exists("./chroma_db"):
    # 直接加载已有数据库
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    print("加载已有数据库")
else:
    # 第一次才创建
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("创建新数据库")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # 存到本地
)

print(f"已存入{vectorstore._collection.count()}个向量")


# ============================================================
# 示例5：相似度检索（重点）
# ============================================================

print("\n" + "=" * 50)
print("示例5：相似度检索")
print("=" * 50)

queries = [
    "什么是RAG？",
    "有哪些向量数据库？",
    "大模型有哪些代表？"
]

for query in queries:
    print(f"\n问题：{query}")
    results = vectorstore.similarity_search(query, k=2)  # 返回最相关的2块
    for i, result in enumerate(results):
        print(f"  相关片段{i+1}：{result.page_content[:50]}...")


# ============================================================
# 示例6：把检索结果喂给模型（RAG雏形）
# ============================================================

print("\n" + "=" * 50)
print("示例6：RAG雏形——检索+生成")
print("=" * 50)

def rag_answer(question):
    # 第一步：检索相关片段
    relevant_chunks = vectorstore.similarity_search(question, k=2)
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])
    
    # 第二步：把片段塞进prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        你是一个问答助手。
        请根据以下参考内容回答用户问题。
        如果参考内容里没有相关信息，就说不知道，不要编造。
        
        参考内容：
        {context}
        """),
        ("user", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    
    return answer, relevant_chunks

# 测试
test_questions = [
    "RAG是什么？",
    "常用的向量数据库有哪些？",
    "Python怎么学？"  # 文档里没有这个信息
]

for q in test_questions:
    print(f"\n问题：{q}")
    answer, chunks = rag_answer(q)
    print(f"回答：{answer}")
    print(f"参考了{len(chunks)}个片段")