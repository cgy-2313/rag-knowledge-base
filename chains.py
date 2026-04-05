from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="sk-b99631e460b0473da7bcce584675814d",
    base_url="https://api.deepseek.com",
    temperature=0
)

parser = StrOutputParser()  # 把模型输出转成纯字符串


# ============================================================
# 示例1：最基础的Chain
# ============================================================

print("=" * 50)
print("示例1：最基础的Chain")
print("=" * 50)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的翻译助手"),
    ("user", "将以下内容翻译成英文：{text}")
])

# 用 | 符号把步骤串起来，这是LangChain的核心语法
chain = prompt | llm | parser

result = chain.invoke({"text": "我正在学习大模型开发"})
print(f"翻译结果：{result}")


# ============================================================
# 示例2：多步骤Chain（重点）
# ============================================================

print("\n" + "=" * 50)
print("示例2：多步骤Chain——总结然后翻译")
print("=" * 50)

# 第一步：总结
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个文章总结助手"),
    ("user", "用一句话总结以下文章：\n{article}")
])

# 第二步：翻译
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个翻译助手"),
    ("user", "将以下内容翻译成英文：\n{summary}")
])

# 手动串联两个步骤
article = """
    大模型技术在2024年迎来了爆发式增长，越来越多的企业开始将大模型
    应用到实际业务中。从客服机器人到代码生成，从文档分析到智能搜索，
    大模型正在改变各行各业的工作方式。对于开发者来说，掌握大模型应用
    开发技能已经成为职场竞争力的重要组成部分。
"""

# 第一步
summary_chain = summary_prompt | llm | parser
summary = summary_chain.invoke({"article": article})
print(f"第一步总结：{summary}")

# 第二步
translate_chain = translate_prompt | llm | parser
translation = translate_chain.invoke({"summary": summary})
print(f"第二步翻译：{translation}")


# ============================================================
# 示例3：PromptTemplate的变量用法（重点）
# ============================================================

print("\n" + "=" * 50)
print("示例3：PromptTemplate变量用法")
print("=" * 50)

# 模板里用{变量名}占位
review_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}"),
    ("user", """
    请对以下{product}的评论做情感分析：
    评论：{review}
    
    只输出：正面、负面、或中性
    """)
])

chain = review_prompt | llm | parser

# 传入多个变量
result = chain.invoke({
    "role": "专业的电商分析师",
    "product": "耳机",
    "review": "音质还不错，但是佩戴不舒适，而且价格偏贵"
})
print(f"情感分析结果：{result}")


# ============================================================
# 示例4：批量处理（实际项目常用）
# ============================================================

print("\n" + "=" * 50)
print("示例4：批量处理多条评论")
print("=" * 50)

reviews = [
    {"role": "电商分析师", "product": "手机", "review": "拍照效果很好，电池续航强"},
    {"role": "电商分析师", "product": "耳机", "review": "音质差，做工粗糙，不值这个价"},
    {"role": "电商分析师", "product": "键盘", "review": "手感一般，价格合理，凑合能用"},
]

# batch方法批量处理
results = chain.batch(reviews)

for i, (review, result) in enumerate(zip(reviews, results)):
    print(f"评论{i+1}（{review['product']}）：{result}")