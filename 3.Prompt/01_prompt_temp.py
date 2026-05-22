from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

topic = input('Research topic\n')
tone = input('tone like easy, techincal, noob, etc\n')

template = PromptTemplate(
    template= """
    You are a researcher, your research topic is "{topic}".
    You donot have to hallicunate and give wrong information. 
    You answer about the topic in a {tone} way. 
    Make sure to be polite and answer in best way without giving false information.
""",
input_variables= ['topic', 'tone'],
validate_template= True
)

prompt = template.invoke({
    'topic': topic, 
    'tone': tone
})

print(model.invoke(prompt).content)
