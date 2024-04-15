import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

checkpoint = 'LaMini-T5-738M'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map='cpu',
    torch_dtype = torch.float32
).to('cpu')

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        max_length=256,
        tokenizer=tokenizer,
        do_sample=True,
        temperature= 0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True
    )
    return qa


def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text


def main():
    st.title("Search your fucking pdf mate!!")
    with st.expander("About the app:"):
        st.text("something something")
    
    question = st.text_area('Enter your question')
    if st.button('Search'):
        st.info("Your question: "+question)
        st.info("Your answer: ")
        answer, metadata = process_answer(question)
        st.write(answer)
        st.write(metadata)

if __name__=='__main__':
    main()
