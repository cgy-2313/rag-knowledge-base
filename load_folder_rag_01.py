import os
import shutil
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)


# ============================================================
# 核心改动：加载整个文件夹
# ============================================================

def load_documents_from_folder(folder_path):
    all_docs = []
    supported_extensions = [".txt", ".pdf"]

    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_extensions:
            continue

        file_path = os.path.join(folder_path, filename)

        try:
            if ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif ext == ".pdf":
                loader = PyPDFLoader(file_path)

            docs = loader.load()
            all_docs.extend(docs)
            print(f"已加载：{filename}，{len(docs)}页，"
                  f"{sum(len(d.page_content) for d in docs)}字符")

        except Exception as e:
            print(f"加载失败：{filename}，原因：{e}")

    print(f"\n共加载{len(all_docs)}个文档页")
    return all_docs


def build_knowledge_base(folder_path, db_path="./rag_db"):
    if os.path.exists(db_path):
        print("检测到已有知识库，直接加载...")
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        print(f"加载完成，共{vectorstore._collection.count()}个片段")
        return vectorstore

    print("构建新知识库...")

    # 加载整个文件夹
    docs = load_documents_from_folder(folder_path)

    if not docs:
        raise ValueError(f"文件夹{folder_path}里没有可加载的文件")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    chunks = splitter.split_documents(docs)
    print(f"切片完成：{len(chunks)}个片段")

    # 打印每个片段的来源，验证多文件加载成功
    print("\n片段来源分布：")
    sources = {}
    for chunk in chunks:
        source = os.path.basename(chunk.metadata.get("source", "未知"))
        sources[source] = sources.get(source, 0) + 1
    for source, count in sources.items():
        print(f"  {source}：{count}个片段")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    print(f"\n知识库构建完成，共{vectorstore._collection.count()}个向量")
    return vectorstore


def rag_answer(question, vectorstore):
    total = vectorstore._collection.count()
    relevant_chunks = vectorstore.similarity_search(question, k=total)

    context = "\n\n".join([
        f"片段{i+1}（来源：{os.path.basename(chunk.metadata.get('source', '未知'))}）："
        f"{chunk.page_content}"
        for i, chunk in enumerate(relevant_chunks)
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个公司内部知识库助手。
请根据以下参考内容回答员工的问题。

规则：
1. 只根据参考内容回答，不要编造信息
2. 如果参考内容里没有相关信息，直接说"知识库中没有相关信息"
3. 回答要简洁清晰
4. 回答末尾标注答案来自哪个文件

参考内容：
{context}"""),
        ("user", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, relevant_chunks


def run_rag_chatbot():
    print("=" * 50)
    print("构建知识库中...")

    if os.path.exists("./rag_db"):
        shutil.rmtree("./rag_db")

    vectorstore = build_knowledge_base("./docs")

    print("\n" + "=" * 50)
    print("知识库问答系统启动")
    print("输入'退出'结束")
    print("=" * 50)

    while True:
        question = input("\n你的问题：").strip()
        if question == "退出":
            print("再见！")
            break
        if not question:
            continue

        print("\n检索中...")
        answer, chunks = rag_answer(question, vectorstore)
        print(f"\n回答：\n{answer}")
        print(f"\n参考片段来源：")
        for chunk in chunks:
            source = os.path.basename(chunk.metadata.get("source", "未知"))
            print(f"  {source}：{chunk.page_content[:30]}...")


if __name__ == "__main__":
    run_rag_chatbot()