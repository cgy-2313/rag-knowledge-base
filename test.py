from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage 

# 初始化模型，和之前一样只改base_url和api_key
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="sk-b99631e460b0473da7bcce584675814d",
    base_url="https://api.deepseek.com",
    temperature=0
)


# ============================================================
# 示例1：最基础的调用
# ============================================================

# response = llm.invoke("你好，请用一句话介绍你自己")
# print("=" * 50)
# print("示例1：最基础的调用")
# print("=" * 50)
# print(response.content)


# ============================================================
# 示例2：带system prompt的调用
# ============================================================

# messages = [
#     SystemMessage(content="你是一个专业的Python编程助手，回答要简洁"),
#     HumanMessage(content="列表和元组的区别是什么？")
# ]

# response = llm.invoke(messages)
# print("\n" + "=" * 50)
# print("示例2：带system prompt")
# print("=" * 50)
# print(response.content)


# ============================================================
# 示例3：多轮对话，对比之前的写法
# ============================================================

messages = [
    SystemMessage(content="你是一个专业的Python编程助手"),
    HumanMessage(content="我在学Python，从哪里开始？"),
]

response = llm.invoke(messages)
print("\n" + "=" * 50)
print("示例3：多轮对话")
print("=" * 50)
print(f"第一轮：{response.content[:50]}...")

# 把回复加入历史，继续对话
messages.append(AIMessage(content=response.content))
messages.append(HumanMessage(content="学完基础之后呢？"))

# print(messages)
response = llm.invoke(messages)
print(f"第二轮：{response.content[:50]}...")


# ============================================================
# 示例4：对比新旧写法（重点理解）
# ============================================================

# print("\n" + "=" * 50)
# print("示例4：新旧写法对比")
# print("=" * 50)

# print("""
# 旧写法（直接调API）：
#   messages = [{"role": "system", "content": "..."}]
#   client.chat.completions.create(messages=messages)

# 新写法（LangChain）：
#   messages = [SystemMessage(content="...")]
#   llm.invoke(messages)

# 本质完全一样，只是LangChain用了更清晰的类来表示角色：
#   SystemMessage  →  role: system
#   HumanMessage   →  role: user  
#   AIMessage      →  role: assistant
# """)