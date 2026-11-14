import os
from litellm import completion
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 里的 DEEPSEEK_API_KEY

response = completion(
    model="deepseek/deepseek-v4-pro",  # 关键：加 deepseek/ 前缀
    messages=[
        {"role": "user", "content": "你好，介绍一下你自己"}
    ]
)

print(response.choices[0].message.content)
