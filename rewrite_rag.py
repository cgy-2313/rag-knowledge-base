import os
import shutil
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = ChatOpenAI(
    model = "deepseek-chat",
    api_key = "sk-b99631e460b0473da7bcce584675814d",
    base_url ="https://api.deepseek.com",
    temperature = 0
)

embedding = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key="sk-htuzymdmwbisppkrndpwyyjdnvxmqhebkavhbfurjjacjnby",
    base_url="https://api.siliconflow.cn/v1"
)

# ============================================================
# 第一步：构建知识库
# ============================================================

def build_knowledge_base(file_path, db_path = "./rag_db"):
    # 如果数据已存在，直接加载
    if(os.path.exists(db_path)):
        print("检测到已有知识库，直接加载...")
        vectorstore = Chroma(
            persist_directory = db_path,
            embedding_function=embedding
        )
        print(f"记载完成，共{vectorstore._collection.count()}个片段")
        return vectorstore
    
    print("构建知识库...")

    # 加载文档
    loader = TextLoader(file_path, encoding = "utf-8")
    docs = loader.load()
    print(f"记载文档{len(docs)}个文件")

    # 切片
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = 20
    )
    chunks = splitter.split_documents(docs)
    print(f"切片完成：{len(chunks)}个片段")

    # 向量化存入Chroma
    vectorstore = Chroma.from_documents(
        persist_directory=db_path,
        embedding=embedding,
        documents=chunks
    )

    print(f"知识库构建完成，共{vectorstore._collection.count()}个向量")

    return vectorstore

# ============================================================
# 第二步：检索+生成
# ============================================================
def rag_answer(question, vectorstore, k=7):
    # 检索最相关的k个片段
    relevant_chunks = vectorstore.similarity_search(question, k=k)

    # 拼接上下文
    context = "\n\n".join([
        f"片段{i+1}：{chunk.page_content}"
        for i, chunk in enumerate(relevant_chunks)
    ])

    # 加这行，看看实际传给模型的内容
    print(f"\n===传给模型的context===\n{context}\n===结束===\n")

    # 构建prompt
    prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个公司内部知识库助手。
请根据以下参考内容回答员工的问题。

规则：
1. 只根据参考内容回答，不要编造信息
2. 如果参考内容里没有相关信息，直接说"知识库中没有相关信息"
3. 回答要简洁清晰，可以用数字列表
4. 回答末尾注明参考了哪些片段

参考内容：
{context}"""),
    ("user", "{question}")
])

    chain = prompt | llm | StrOutputParser()
    
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer, relevant_chunks

# ============================================================
# 第三步：完整对话界面
# ============================================================

def run_rag_chatbot():
    print("=" * 50)
    print("构建知识库中...")
    vectorstore = build_knowledge_base("knowledge_base.txt")

    print("\n" + "=" * 50)
    print("知识库问答系统启动")
    print("输入'退出'结束")
    print("=" * 50)

    while True:
        question = input("\n你的问题：").strip()

        if question == "退出":
            print("再见!")
            break

        if not question:
            continue

        print("\n检索中...")
        answer, chunks = rag_answer(question,vectorstore) 

        print(f"\n回答：{answer}")
        print(f"检索了{len(chunks)}个相关片段")

# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    if os.path.exists("./rag_db"):
        shutil.rmtree("./rag_db")
        print("已删除旧数据库")
    run_rag_chatbot()