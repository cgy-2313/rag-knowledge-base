from dotenv import load_dotenv
import shutil
import os
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

# 然后正常使用
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0
)

embedding = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=SILICONFLOW_API_KEY,
    base_url="https://api.siliconflow.cn/v1"
)

# ============================================================
# 核心改动：根据文件类型自动选择加载器
# ============================================================

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        print(f"使用PDF加载器：{file_path}")
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
        print(f"使用TXT加载器：{file_path}")
    else:
        raise ValueError(f"不支持的文件类型：{ext}")
    
    docs = loader.load()
    print(f"加载完成：{len(docs)}页，共{sum(len(d.page_content) for d in docs)}字符")
    return docs

def build_knowledge_base(file_path, db_path="./rag_db"):
    if os.path.exists(db_path):
        print("检测到已有知识库，直接加载ing")
        vectorstore = Chroma(
            embedding_function=embedding,
            persist_directory=db_path
        )
        print(f"加载完成，共{vectorstore._collection.count()}个片段")
        return vectorstore
    
    print("构建新知识库...")

    # 用新的load_document函数加载
    docs = load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    chunks = splitter.split_documents(docs)
    print(f"切片完成：{len(chunks)}个片段")

    # 打印每个片段的元数据，PDF会包含页码
    print("\n片段元数据示例：")
    for i,chunk in enumerate(chunks):
        print(f"片段{i+1}：{chunk.metadata}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        persist_directory=db_path,
        embedding=embedding
    )
    
    print(f"知识库构建完成，共{vectorstore._collection.count()}个片段")
    return vectorstore

def rag_answer(question, vectorstore, k=8):
    relevant_chunks = vectorstore.similarity_search(question, k=k)

    context = "\n\n".join([
        f"片段{i+1}（来源：{chunk.metadata}）：{chunk.page_content}"
        for i, chunk in enumerate(relevant_chunks)
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system","""
你是一个公司内部知识库助手。
请根据以下参考内容回答员工的问题。

规则：
1.只根据参考内容回答，不要编造信息
2.如果参考内容没有相关信息，直接说"知识库中没有相关信息"
3.回答要简洁清晰
4.问题末尾注明答案来自哪个来源
         
参考内容：
{context}
         """),
        ("user", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, relevant_chunks


def run_rag_chatbot():
    print("=" * 50)
    print("构建知识库中...")
    
    # 改这里：可以传入pdf或txt
    vectorstore = build_knowledge_base("knowledge_base.pdf")

    print("\n" + "=" * 50)
    print("知识库问答系统启动")
    print("输入'退出'结束")
    print("=" * 50)

    while True:
        question = input("\n输入你的问题：").strip()
        if question == "退出":
            print("再见！")
            break
        if not question:
            continue

        print("\n检索中...")
        answer, chunks = rag_answer(question, vectorstore)
        print(f"\n回答：\n{answer}")
        print(f"\n[检索了{len(chunks)}个相关片段]")

if __name__ == "__main__":
    # 先清空旧数据库
    if os.path.exists("./rag_db"):
        shutil.rmtree("./rag_db")

    run_rag_chatbot()
    
