from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt

load_dotenv()

model = ChatOpenAI()

template = load_prompt('prompt1.json')

chain = template | model

res = chain.invoke({
        "tone" : "Technoical but easy",
        "topic" : "MCP"
})

print(res.content)