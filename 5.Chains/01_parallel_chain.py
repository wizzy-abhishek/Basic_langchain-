from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(template= """
You have to summarize the notes in such a way that students find easy to understand the topic.
                         Keep it technical, but easy to understand. 
                         Summarize in such a way that if anyone reads it, they can give test on that notes. 
                         Note:- {note}
""", input_variables=['note'])

prompt2 = PromptTemplate(template= """
You have to generate 5 - 10 question from the provided notes. 
                         If you keep 5 question make sure it is hard to medium.
                         If it is more than five keep it easy to hard. 
                         Like if number of question increases the difficulty decrease. 
                         However, make sure wheather there is 5, 6, 7, 8, 9, or 10 question. The overall difficulty remains the same. 
                         Note:- {note}
""", input_variables=['note'])

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model | parser, 
    'quiz' : prompt2 | model | parser
})

prompt3 = PromptTemplate(template="""
You are an assistant organizing study materials. Combine the provided study notes and quiz questions exactly as they are given to you. 
Do not rewrite them, do not turn questions into answers, and do not lose the formatting.

Structure your response EXACTLY like this:
# STUDY NOTES
{notes}

---

# QUIZ QUESTIONS
{quiz}
""", input_variables=['notes', 'quiz'])

mereger_chain = prompt3 | model | parser

complete_chain = parallel_chain | mereger_chain

with open("Extra Data/Notes_for_parallelchain.txt", "r", encoding="utf-8") as file:
    content = file.read()

res = complete_chain.invoke({'note': content})
print(res)