# import
from langchain_chroma import Chroma
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter
import os

# # load the document and split it into chunks
# loader = PyPDFLoader('docs/tmp.pdf')
# documents = loader.load()

# # split it into chunks
# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=500)
# docs = text_splitter.split_documents(documents)

# # create the open-source embedding function
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# # load it into Chroma
# db = Chroma.from_documents(docs, embedding_function, persist_directory="db")



def main():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embedding_function, persist_directory="db")


if __name__=='__main__':
    main()