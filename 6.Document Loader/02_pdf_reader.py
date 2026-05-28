from langchain_community.document_loaders import PyPDFLoader

filepath = 'Extra Data/MCPSystematicReviewV2.pdf'
loader = PyPDFLoader(file_path= filepath)

#docs = loader.load()
#print(len(docs))

docs = loader.lazy_load()

print(type(docs))

for doc in docs:
    print(doc)