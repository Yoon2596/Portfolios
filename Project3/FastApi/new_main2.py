import os
import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
import uvicorn
from ultralytics import YOLO
import torch
import logging

# model_echo.py에서 필요한 함수 임포트
from Model_Echo import analyze_video_and_generate_text

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# main2.py의 상단에 다음 import 추가
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시에는 특정 도메인으로 제한하는 것이 좋습니다
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB 연결 설정
MONGODB_URL = "mongodb+srv://yoonsun2596:qwer1234@tm2.hl7a3.mongodb.net"
# MONGO_URL = os.getenv('MONGO_URL', 'mongodb+srv://ihyuns96:qwer1234@cluster0.xakad.mongodb.net/')
client = AsyncIOMotorClient(MONGODB_URL)
db = client.get_database("LOG")
collection = db.get_collection("chat_log")

# 모델 실행을 위한 설정 (필요에 따라 수정)
YOLO_MODEL = YOLO(r"/Users/bagjihwan/Desktop/flutter/fonts/yolo_model.pt")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONF_THRESH = 0.5  # 신뢰도 임계값
CHECKPOINT = torch.load(
    r'/Users/bagjihwan/Desktop/flutter/fonts/lstm_model.pt',
    map_location=DEVICE
)
FONT_PATH = r"E:/1014/malgun.ttf"  # 한글 폰트 파일 경로

# 분석 결과를 저장할 Pydantic 모델
class AnalysisResult(BaseModel):
    yolo_text: str
    lstm_text: str
    summary_text: str
    rag_text: str
    follow_up_answer: str
    upload_time: datetime.datetime

@app.post("/upload_video/", response_model=AnalysisResult)
async def upload_video(file: UploadFile = File(...)):
    import shutil  # 파일 저장을 위한 임포트 추가

    try:
        # 파일 저장
        file_location = f"uploaded_videos/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"파일 저장 완료: {file_location}")

        # AI 분석 수행
        analysis_results = await analyze_video_and_generate_text(
            video_path=file_location,
            yolo_model=YOLO_MODEL,
            device=DEVICE,
            conf_thresh=CONF_THRESH,
            checkpoint=CHECKPOINT,
            font_path=FONT_PATH
        )

        logger.info("AI 분석 완료")
        

        document = {
            "filename": file.filename,
            "upload_time": datetime.datetime.utcnow(),
            "yolo_text": analysis_results.get("yolo_text"),
            "lstm_text": analysis_results.get("lstm_text"),
            "summary_text": analysis_results.get("summary_text"),
            "rag_text": analysis_results.get("rag_text"),
            "follow_up_answer": analysis_results.get("follow_up_answer"),
            "chat_history": []
        }



        # MongoDB에 문서 삽입 또는 업데이트 (upsert 사용)
        result = await collection.update_one(
            {"email": current_user},
            {"$set": {"result": document}},
            upsert=True
        )


        if result.upserted_id:
            logger.info(f"MongoDB에 데이터 삽입 완료: {result.upserted_id}")
            session_id = str(result.upserted_id)
        else:
            logger.info(f"MongoDB에 데이터 업데이트 완료: {current_user}")
            session_id = current_user

        # 세션 ID 생성
        sessions[session_id] = {
            "analysis_results": analysis_results
        }

        return AnalysisResult(
            email=current_user,
            result={
                "yolo_text": document["yolo_text"],
                "lstm_text": document["lstm_text"],
                "summary_text": document["summary_text"],
                "rag_text": document["rag_text"],
                "follow_up_answer": document["follow_up_answer"],
            }
        )
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")


# # Pydantic 모델 정의
# class ChatHistoryRequest(BaseModel):
#     user_email: str

# class ChatHistoryUpdateRequest(BaseModel):
#     user_email: str
#     user_question: str
#     assistant_answer: str

async def get_session_id_from_request(request: Request):
    body = await request.json()
    email = body.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="이메일이 요청 본문에 없습니다.")
    
    document = await collection.find_one({"email": email})
    if not document:
        raise HTTPException(status_code=404, detail="해당 이메일을 가진 세션을 찾을 수 없습니다.")
    return document["_id"]  # session_id를 반환


    
# def get_chat_history(user_email: str):
#     user_data = collection.find_one({"email": user_email})
#     if user_data and "chat_history" in user_data:
#         return user_data["chat_history"]
#     return []

# def update_chat_history(user_email: str, user_question: str, assistant_answer: str):
#     collection.update_one(
#         {"email": user_email},
#         {"$push": {"chat_history": {"question": user_question, "answer": assistant_answer}}}
#     )



@app.post("/ask")
async def ask_question(request: QuestionRequest, session_id: str = Depends(get_session_id_from_request)):
    try: # 가져오는거 확인
        print(f"Received question: {request.question}")
        print(f"Session ID: {session_id}")

        if request.question.strip().lower() == "exit":
            del sessions[session_id]
            await collection.delete_one({"_id": session_id})
            return {"message": "세션이 종료되었습니다."}

        session = sessions.get(session_id)
        if not session:
            # 세션이 메모리에 없을 경우 MongoDB에서 불러오기 시도
            document = await collection.find_one({"_id": session_id})
            if not document:
                raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
            sessions[session_id] = {
                "chat_history": document.get("chat_history", []),
                "analysis_results": {
                    "yolo_text": document.get("yolo_text"),
                    "lstm_text": document.get("lstm_text"),
                    "summary_text": document.get("summary_text"),
                    "rag_text": document.get("rag_text"),
                    "follow_up_answer": document.get("follow_up_answer")
                }
            }
            session = sessions[session_id]
            print(session)

        # LLM 응답 생성
        user_answer = await handle_user_question(
            user_question=request.question,
            chat_history=session["chat_history"],
            analysis_results=session["analysis_results"]
        )


        # # 대화 내역에 사용자의 질문 추가
        # session["chat_history"].append({"role": "user", "content": request.question})
        # # 대화 내역에 LLM의 응답 추가
        # session["chat_history"].append({"role": "assistant", "content": user_answer['answer']})

        # 대화 내역에 사용자의 질문과 LLM의 응답을 하나의 객체로 저장
        session["chat_history"].append({
            "role": "user",
            "content": request.question,
            "assistant_response": user_answer['answer']
        })
        # MongoDB에 대화 기록 업데이트
        result = await collection.update_one(
            {"_id": session_id},
            {"$set": {
                "chat_history": session["chat_history"]
            }}
        )

        if result.upserted_id:
            logger.info(f"MongoDB에 데이터 삽입 완료: {result.upserted_id}")
        else:
            logger.info(f"MongoDB에 데이터 업데이트 완료: {session_id}")

        # 유저 질문에 대한 대답 
        print(user_answer['answer'])
        # return {"answer": user_answer['answer']}
        return JSONResponse(content=user_answer, media_type="application/json; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

