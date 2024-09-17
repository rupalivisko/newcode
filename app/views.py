import json
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import JsonResponse
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA

pdf_path = "C:/Users/pc/Downloads/Updated Remark App Description.pdf"
google_api_key = 'AIzaSyAJLv_QjBn1QPliUJ6_CTR4peHzd2cXVYg'
embedding_model_path = "models/embedding-001"

def initialize_chatbot(pdf_path, google_api_key, embedding_model_path):
    model = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash-latest',
        temperature=0.7,
        google_api_key=google_api_key
    )
    embedding_model = GoogleGenerativeAIEmbeddings(
        model=embedding_model_path,
        google_api_key=google_api_key
    )

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    doc_search = DocArrayInMemorySearch.from_documents(splits, embedding_model)

    rag_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=doc_search.as_retriever(),
        chain_type="stuff"
    )
    
    return rag_chain

rag_chain = initialize_chatbot(pdf_path, google_api_key, embedding_model_path)

def chat_with_remark(rag_chain, user_input):
    response = rag_chain.run(query=user_input)
    response_text = ' '.join(response.split())  
    response_text = response_text.replace('\n', ' ').strip()
    response_text = response_text.replace('*', '')    
    response_text = response_text.replace('• ', '')  

    return response_text

@csrf_exempt
def Chatbot(request):
    if request.method == 'GET':
        user_input = request.GET.get('q', '')  
        if user_input:
            response = chat_with_remark(rag_chain, user_input)
            return JsonResponse({'response': response})
        return JsonResponse({'error': 'No query parameter provided'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def home(request):
    return render(request, 'home.html')
