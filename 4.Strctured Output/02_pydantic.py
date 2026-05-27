from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Optional, Annotated
from langchain_core.prompts import PromptTemplate

class Review(BaseModel):

    sentiment: Annotated[str, """Return the sentiment of the text provided. 
                        Say positive, negative, or neutal. 
                        Give output in these 3 expression only.
                        Exactly like that don't start with a capital letter.
                        Just give one-word answer.  
                        """]
    
    additional_information : Optional[list]
load_dotenv()
model = ChatOpenAI()

prompt = PromptTemplate(template="""You are an AI assitant that classifies the text into feedback type segment. 

                                    review = {review}

                                    """, input_variables=['review'])


struc_model = model.with_structured_output(Review)

chain = prompt | struc_model

input = """The content is too vague and not detailed enough. 
Try expanding each point and including definitions to make it clearer."""

res = chain.invoke({'review': input})
res = dict(res)
print(res['sentiment'])

# We can add extra validation with Pydantic and Field
