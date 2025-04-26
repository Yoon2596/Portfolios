# retriever.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain.retrievers import MergerRetriever
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from IPython.display import Image, display, Audio
from pymongo import MongoClient
import pprint
import os

os.environ['OPENAI_API_KEY'] = 'API 키 필요'
# MongoDB 연결 설정
uri = "mongodb+srv://yoonsun2596:qwer1234@tm2.hl7a3.mongodb.net/"
db_client = MongoClient(uri)
db = db_client['your_database_name']

def get_ret():
    # 임베딩 모델 설정
    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs
    )
    
    # MongoDB Atlas에서 첫 번째 벡터 스토어 불러오기
    index_name = 'vector_index_1'
    dbName = "VectorStore_RAG_54"
    collectionName = "RAG_traffic_accidents_54"
    collection = db_client[dbName][collectionName]

    vectorStore1 = MongoDBAtlasVectorSearch(
        embedding=embeddings_model,
        collection=collection,
        index_name=index_name
    )

    # general json 자료
    index_name_json = 'general_index'
    dbName_json = "dbsparta"
    collectionName_json = "general_json"
    collection_json = db_client[dbName_json][collectionName_json]

    vectorStore_json1 = MongoDBAtlasVectorSearch(
        embedding=embeddings_model,
        collection=collection_json,
        index_name=index_name_json,
        embedding_key="vector",  # JSON 문서에서 벡터 필드의 키
        text_key="accidentDetails"  # JSON 문서에서 텍스트 필드의 키
    )

    # if json 자료
    index_name_json = 'if_index'
    dbName_json = "dbsparta"
    collectionName_json = "if_json"
    collection_json = db_client[dbName_json][collectionName_json]

    vectorStore_json2 = MongoDBAtlasVectorSearch(
        embedding=embeddings_model,
        collection=collection_json,
        index_name=index_name_json,
        embedding_key="vector",  # JSON 문서에서 벡터 필드의 키
        text_key="accidentOverview"  # JSON 문서에서 텍스트 필드의 키
    )

    # 각 vectorstore에서 retriever를 생성
    retriever1 = vectorStore1.as_retriever()
    retriever_json1 = vectorStore_json1.as_retriever()
    retriever_json2 = vectorStore_json2.as_retriever()
    
    # MergerRetriever를 사용하여 모든 검색기 통합
    merger_retriever = MergerRetriever(retrievers=[retriever1, retriever_json1, retriever_json2])
    
    return merger_retriever

def rag_qa(query, k=3):
    retriever = get_ret()
    try:
        # MergerRetriever는 get_relevant_documents 메소드를 사용합니다
        documents = retriever.get_relevant_documents(query)
        
        # 점수를 기준으로 정렬할 수 없으므로, 단순히 상위 k개의 문서를 선택합니다
        search_results = documents[:k]
        
        # 결과를 딕셔너리 형태로 변환
        formatted_results = [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "animationURL": doc.metadata.get('animationURL', '')
            } for doc in search_results
        ]
        print("RAG QA Result:", formatted_results)  # 디버깅 로그
        
        return formatted_results
    except Exception as e:
        print(f"검색 중 오류가 발생했습니다: {str(e)}")
        return []
