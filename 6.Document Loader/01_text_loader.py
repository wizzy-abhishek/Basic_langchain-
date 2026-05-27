from langchain_community.document_loaders import TextLoader

filepath = "Extra Data/Notes_for_parallelchain.txt"
loader = TextLoader(file_path=filepath , autodetect_encoding= True)

docs = loader.load()

print(len(docs)) #Loads the whole document 
print(docs[0].page_content)
print(type(docs))