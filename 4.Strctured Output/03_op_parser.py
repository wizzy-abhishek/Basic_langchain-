from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()
model = ChatOpenAI()

prompt = PromptTemplate(template="""You are an AI assitant that classifies the text into feedback type segment. 
                                    Return the sentiment of the text provided. 
                                    Say positive, negative, or neutal. 
                                    Give output in these 3 expression only.
                                    Exactly like that don't start with a capital letter.
                                    Just give one-word answer. 
                                    review = {review}

                                    """, input_variables=['review'])

parser = StrOutputParser()

chain = prompt | model | parser

input = """
        You have organized the content well, but some parts are unclear and lack depth. 
        Adding more detailed explanations and examples will improve it.
"""

res = chain.invoke({'review':input})

print(res)
