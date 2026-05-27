from typing import TypedDict, Annotated
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

class Review(TypedDict):

    sentiment: Annotated[str, """Return the sentiment of the text provided. 
                        Say positive, negative, or neutal. 
                        Give output in these 3 expression only.
                        Exactly like that don't start with a capital letter.
                        Just give one-word answer.  
                        """]
    
load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(template="""You are an AI assitant that classifies the text into feedback type segment. 

                                    review = {review}

                                    """, input_variables=['review'])

struc_model = model.with_structured_output(Review)

chain = prompt | struc_model

input = """Your structure is good and easy to read. 
To improve further, try adding real-world examples and a bit more explanation in complex sections."""

res = chain.invoke({'review': input})
print(res)