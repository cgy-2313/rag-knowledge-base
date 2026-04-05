from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from openai import OpenAI

api_key = "sk-b99631e460b0473da7bcce584675814d"
base_url = "https://api.deepseek.com"

# LangChain写法
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url=base_url,
    temperature=0
)

messages = [
    SystemMessage(content="你是一个助手"),
    HumanMessage(content="用一句话解释什么是API")
]

response_langchain = llm.invoke(messages)

# 原始API写法
client = OpenAI(api_key=api_key, base_url=base_url)

response_raw = client.chat.completions.create(
    model="deepseek-chat",
    temperature=0,
    messages=[
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "用一句话解释什么是API"}
    ]
)

# 对比两个结果
print("LangChain结果：")
print(response_langchain.content)

print("\n原始API结果：")
print(response_raw.choices[0].message.content)

print("\n两者返回的内容是否一致：")
print(response_langchain.content == response_raw.choices[0].message.content)