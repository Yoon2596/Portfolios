import os
import datetime
import shutil
import logging
import requests
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from motor.motor_asyncio import AsyncIOMotorClient
from jose import JWTError, jwt
import bcrypt
import torch
from ultralytics import YOLO
from datetime import timedelta, datetime as dt
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_401_UNAUTHORIZED
from fastapi.responses import JSONResponse

# model_echo.py에서 필요한 함수 임포트
from Model_Echo_entire import analyze_video_and_generate_text, handle_user_question


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# MongoDB 설정
# MONGO_URL = os.getenv('MONGO_URL', 'mongodb+srv://ihyuns96:qwer1234@cluster0.xakad.mongodb.net/')
# client = AsyncIOMotorClient(MONGO_URL)
# db = client['mydatabase']
# collection = db['users']
# chat_log_collection = db.get_collection("chat_log")

MONGODB_URL = "mongodb+srv://yoonsun2596:qwer1234@tm2.hl7a3.mongodb.net"
client = AsyncIOMotorClient(MONGODB_URL)
db = client.get_database("LOG")
collection = db.get_collection("video_log")

# 모델 실행을 위한 설정 (필요에 따라 수정)
YOLO_MODEL = YOLO(r"D:\kdt_240424\workspace\project3\flutterlogin\fonts\yolo_model.pt")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONF_THRESH = 0.5  # 신뢰도 임계값
CHECKPOINT = torch.load(
    r'D:\kdt_240424\workspace\project3\flutterlogin\fonts\lstm_model.pt',
    map_location=DEVICE
)
FONT_PATH = r"D:\kdt_240424\workspace\project3\flutterlogin\fonts\malgun.ttf"  # 한글 폰트 파일 경로





# 세션 별 대화 내역을 저장
sessions = {}

class QuestionRequest(BaseModel):
    email: EmailStr
    question: str

# FastAPI 앱 생성
app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic 모델 정의
class User(BaseModel):
    email: EmailStr
    password: str

# class AnalysisResult(BaseModel):
#     yolo_text: str
#     lstm_text: str
#     summary_text: str
#     rag_text: str
#     follow_up_answer: str
#     upload_time: datetime.datetime

class AnalysisResult(BaseModel):
    email: str
    result: dict


# 비밀번호 암호화 함수 (bcrypt 사용)
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# JWT 설정
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 토큰 생성 함수
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = dt.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# 비밀번호 확인 함수
# def verify_password(plain_password, hashed_password):
#     return bcrypt.checkpw(plain_password.encode(), hashed_password) 

# # OAuth2PasswordBearer를 사용하여 토큰을 가져오는 방법을 정의합니다.
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# # 사용자 인증을 위한 의존성 함수
# async def get_current_user(token: str = Depends(oauth2_scheme)):
#     credentials_exception = HTTPException(
#         status_code=HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         email: str = payload.get("sub")
#         if email is None:
#             raise credentials_exception
#     except JWTError:
#         raise credentials_exception
    
#     print(email)
#     return email


# 테스트용
async def get_current_user(token: str = "b@b.com"):  # 하드코딩된 이메일
    credentials_exception = HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # JWT 토큰 검증 생략
    email = token  # 하드코딩된 이메일을 사용
    
    if email is None:
        raise credentials_exception
    
    print(email)  # 하드코딩된 이메일 출력
    return email

@app.post("/signup")
async def create_user(user: User):
    try:
        existing_user = await collection.find_one({"email": user.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="이미 존재하는 이메일입니다")

        hashed_password = hash_password(user.password)
        new_user = {
            "email": user.email,
            "password": hashed_password
        }
        result = await collection.insert_one(new_user)

        return {"id": str(result.inserted_id), "message": "사용자가 성공적으로 생성되었습니다"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")


@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        print("로그인 함수 호출됨")  # 로그 출력
        user = await collection.find_one({"email": form_data.username})
        if not user:
            print("사용자 없음")  # 로그 출력
            raise HTTPException(status_code=400, detail="이메일 또는 비밀번호가 잘못되었습니다")
        if not verify_password(form_data.password, user["password"]):
            print("비밀번호 불일치")  # 로그 출력
            raise HTTPException(status_code=400, detail="이메일 또는 비밀번호가 잘못되었습니다")
        
        access_token = create_access_token(data={"sub": user["email"]})
        print(f"로그인 성공: {user['email']}")  # 로그 출력
        
        return {"access_token": access_token, "token_type": "bearer"}
        
    except Exception as e:
        print(f"로그인 중 오류 발생: {str(e)}")  # 오류 메시지 출력
        raise HTTPException(status_code=500, detail="서버 오류")


# @app.post("/save_analysis_result")
# async def save_analysis_result(analysis: AnalysisResult):
#     try:
#         result = await db['analysis_results'].insert_one({
#             "email": analysis.email,
#             "result": analysis.result
#         })
#         return {"message": "Analysis result saved successfully", "id": str(result.inserted_id)}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")


@app.post("/upload", response_model=AnalysisResult)
async def upload_video(
    file: UploadFile = File(...), 
    current_user: str = Depends(get_current_user)):  # 현재 로그인된 사용자 이메일 가져오기
    try:
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

