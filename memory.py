from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="sk-b99631e460b0473da7bcce584675814d",
    base_url="https://api.deepseek.com",
    temperature=0
)


# ============================================================
# 示例1：用列表手动管理Memory（新版推荐写法）
# ============================================================

print("=" * 50)
print("示例1：自动维护对话历史")
print("=" * 50)

messages = [SystemMessage(content="你是一个有帮助的助手")]

def chat(user_input):
    messages.append(HumanMessage(content=user_input))
    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.content))
    return response.content

reply1 = chat("我叫小明，我在学Python")
print(f"第一轮：{reply1}")

reply2 = chat("我叫什么名字？在学什么？")
print(f"第二轮：{reply2}")

print("\n当前messages里存了什么：")
for m in messages:
    print(f"  {type(m).__name__}：{m.content[:30]}...")


# ============================================================
# 示例2：BufferWindowMemory——只记住最近N轮
# ============================================================

print("\n" + "=" * 50)
print("示例2：只记住最近2轮对话")
print("=" * 50)

def chat_with_window(messages, user_input, system_prompt, k=2):
    # 过滤出非system的历史
    history = [m for m in messages if not isinstance(m, SystemMessage)]
    
    # 只保留最近k轮（每轮=human+ai共2条）
    if len(history) > k * 2:
        history = history[-(k * 2):]
    
    # 重新组合：system + 最近k轮历史 + 新消息
    new_input = HumanMessage(content=user_input)
    current_messages = [SystemMessage(content=system_prompt)] + history + [new_input]
    
    response = llm.invoke(current_messages)
    
    # 把新的一轮加入完整历史
    messages.append(new_input)
    messages.append(AIMessage(content=response.content))
    
    return response.content

system = "你是一个有帮助的助手"
window_messages = []

chat_with_window(window_messages, "我叫小明", system, k=2)
chat_with_window(window_messages, "我在学Python", system, k=2)
chat_with_window(window_messages, "我住在广州", system, k=2)  # 此后第一轮被遗忘

reply = chat_with_window(window_messages, "我叫什么名字？", system, k=2)
print(f"问名字（应该不记得）：{reply}")

reply = chat_with_window(window_messages, "我住在哪里？", system, k=2)
print(f"问住址（应该记得）：{reply}")


# ============================================================
# 示例3：SummaryMemory——自动总结压缩历史
# ============================================================

print("\n" + "=" * 50)
print("示例3：自动总结压缩对话历史")
print("=" * 50)

summary = ""  # 存储历史摘要
summary_messages = []

def chat_with_summary(user_input):
    global summary
    
    # 把摘要放进system prompt
    system_content = "你是一个有帮助的助手。"
    if summary:
        system_content += f"\n\n之前对话的摘要：{summary}"
    
    current_messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_input)
    ]
    
    response = llm.invoke(current_messages)
    reply = response.content
    
    # 更新摘要
    summary_messages.append(f"用户：{user_input}")
    summary_messages.append(f"助手：{reply}")
    
    # 每2轮压缩一次
    if len(summary_messages) >= 4:
        compress_prompt = f"""
        请将以下对话压缩成一段简短的摘要，保留关键信息：
        
        {chr(10).join(summary_messages)}
        """
        summary_response = llm.invoke([HumanMessage(content=compress_prompt)])
        summary = summary_response.content
        summary_messages.clear()
        print(f"\n[已压缩，当前摘要]：{summary}\n")
    
    return reply

chat_with_summary("我叫小明，是广州人，深圳大学软件工程专业毕业")
chat_with_summary("我现在在学习大模型开发，目标是找到月薪20k的工作")
chat_with_summary("我之前主要学Java，现在在补Python")
reply = chat_with_summary("帮我总结一下我的情况")
print(f"最终回复：{reply}")