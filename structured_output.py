from pydantic import BaseModel
from litellm import completion

class ExtractedInfo(BaseModel):
    name: str
    email: str
    phone: str | None = None

system_prompt = """You are a helpful assistant that extracts contact information.
Output the result in JSON format with these fields:
- name (string)
- email (string)
- phone (string or null)

Example: {"name": "Jane Doe", "email": "jane@example.com", "phone": "555-9999"}"""

response = completion(
    model="deepseek/deepseek-v4-pro",
    messages=[{
        "role": "user",
        "content": "My name is John Smith, my email is john@example.com, and my phone is 555-1234."
    },
    {"role": "system", "content": system_prompt},
    ],
     response_format={"type": "json_object"}
)

content = response.choices[0].message.content
if content and content.strip():
    result = ExtractedInfo.model_validate_json(content)
    print(result)
else:
    # DeepSeek 在无法生成合法 JSON 时可能返回空白内容[citation:2][citation:5]
    print("模型未返回有效 JSON，可能需要重试或调整 prompt")
