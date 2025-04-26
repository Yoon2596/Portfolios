import json
import os
import base64
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm
from IPython.display import Image
from moviepy.editor import VideoFileClip
from ultralytics import YOLO
from collections import Counter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.retrievers import MergerRetriever
from pymongo.errors import ConnectionFailure
from pymongo import MongoClient
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
import torch.nn as nn
import io
from openai import OpenAI
import logging
import sys
import asyncio
import joblib
from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder



model = "gpt-4o-mini-2024-07-18"
OPENAI_API_KEY = '키 필요'
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)


MONGODB_URI = "mongodb+srv://ihyuns96:qwer1234@cluster0.xakad.mongodb.net/?retryWrites=true&w=majority"
db_client = MongoClient(MONGODB_URI)
db = db_client['dbsparta']
collection = db['dog']

# 한글 폰트 설정
font_path = r"D:\kdt_240424\workspace\project3\flutterlogin\fonts\malgun.ttf"
font_prop = fm.FontProperties(fname=font_path, size=12)
plt.rc('font', family=font_prop.get_name())
plt.rcParams['axes.unicode_minus'] = False 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Model_Echo")

breed_mapping = {
    "Chihuahua": "치와와",
    "Japanese_spaniel": "재패니즈 스패니얼",
    "Maltese_dog": "말티즈",
    "Pekinese": "페키니즈",
    "Shih-Tzu": "시추",
    "Blenheim_spaniel": "블레넘 스패니얼",
    "Papillon": "파피용",
    "Toy_terrier": "토이 테리어",
    "Rhodesian_ridgeback": "로디지안 리지백",
    "Afghan_hound": "아프간 하운드",
    "Basset": "바셋",
    "Beagle": "비글",
    "Bloodhound": "블러드하운드",
    "Bluetick": "블루틱 쿤하운드",
    "Black-and-tan_coonhound": "블랙앤탄 쿤하운드",
    "Walker_hound": "워커 하운드",
    "English_foxhound": "잉글리시 폭스하운드",
    "Redbone": "레드본 하운드",
    "Borzoi": "보르조이",
    "Irish_wolfhound": "아이리시 울프하운드",
    "Italian_greyhound": "이탈리안 그레이하운드",
    "Whippet": "휘핏",
    "Ibizan_hound": "이비전 하운드",
    "Norwegian_elkhound": "노르웨이 엘크하운드",
    "Otterhound": "오터하운드",
    "Saluki": "살루키",
    "Scottish_deerhound": "스코티시 디어하운드",
    "Weimaraner": "바이마라너",
    "Staffordshire_bullterrier": "스태퍼드셔 불테리어",
    "American_Staffordshire_terrier": "아메리칸 스태퍼드셔 테리어",
    "Bedlington_terrier": "베들링턴 테리어",
    "Border_terrier": "보더 테리어",
    "Kerry_blue_terrier": "케리 블루 테리어",
    "Irish_terrier": "아이리시 테리어",
    "Norfolk_terrier": "노퍽 테리어",
    "Norwich_terrier": "노리치 테리어",
    "Yorkshire_terrier": "요크셔 테리어",
    "Wire-haired_fox_terrier": "와이어 폭스 테리어",
    "Lakeland_terrier": "레이클랜드 테리어",
    "Sealyham_terrier": "실리엄 테리어",
    "Airedale": "에어데일",
    "Cairn": "케언 테리어",
    "Australian_terrier": "오스트레일리안 테리어",
    "Dandie_Dinmont": "댄디 딘몬트 테리어",
    "Boston_bull": "보스턴 불",
    "Miniature_schnauzer": "미니어처 슈나우저",
    "Giant_schnauzer": "자이언트 슈나우저",
    "Standard_schnauzer": "스탠다드 슈나우저",
    "Scotch_terrier": "스카치 테리어",
    "Tibetan_terrier": "티베탄 테리어",
    "Silky_terrier": "실키 테리어",
    "Soft-coated_wheaten_terrier": "소프트 코티드 휘튼 테리어",
    "West_Highland_white_terrier": "웨스트 하이랜드 화이트 테리어",
    "Lhasa": "라사압소",
    "Flat-coated_retriever": "플랫 코티드 리트리버",
    "Curly-coated_retriever": "컬리 코티드 리트리버",
    "Golden_retriever": "골든 리트리버",
    "Labrador_retriever": "래브라도 리트리버",
    "Chesapeake_Bay_retriever": "체서피크 베이 리트리버",
    "German_short-haired_pointer": "저먼 쇼트헤어드 포인터",
    "Vizsla": "비즐라",
    "English_setter": "잉글리시 세터",
    "Irish_setter": "아이리시 세터",
    "Gordon_setter": "고든 세터",
    "Brittany_spaniel": "브리타니 스패니얼",
    "Clumber": "클럼버 스패니얼",
    "English_springer": "잉글리시 스프링거 스패니얼",
    "Welsh_springer_spaniel": "웰시 스프링거 스패니얼",
    "Cocker_spaniel": "코커 스패니얼",
    "Sussex_spaniel": "서식스 스패니얼",
    "Irish_water_spaniel": "아이리시 워터 스패니얼",
    "Kuvasz": "쿠바즈",
    "Schipperke": "스키퍼키",
    "Groenendael": "그로넨달",
    "Malinois": "말리노이즈",
    "Briard": "브리아드",
    "Kelpie": "켈피",
    "Komondor": "코몬도르",
    "Old_English_sheepdog": "올드 잉글리시 쉽독",
    "Shetland_sheepdog": "셰틀랜드 쉽독",
    "Collie": "콜리",
    "Border_collie": "보더 콜리",
    "Bouvier_des_Flandres": "부비에 데 플랑드르",
    "Rottweiler": "로트와일러",
    "German_shepherd": "저먼 셰퍼드",
    "Doberman": "도베르만",
    "Miniature_pinscher": "미니어처 핀셔",
    "Greater_Swiss_Mountain_dog": "그레이터 스위스 마운틴 독",
    "Bernese_mountain_dog": "버니즈 마운틴 독",
    "Appenzeller": "아펜젤러",
    "EntleBucher": "엔틀레부처",
    "Boxer": "복서",
    "Bull_mastiff": "불마스티프",
    "Tibetan_mastiff": "티베탄 마스티프",
    "French_bulldog": "프렌치 불도그",
    "Great_Dane": "그레이트 데인",
    "Saint_Bernard": "세인트 버나드",
    "Eskimo_dog": "에스키모 독",
    "Malamute": "말라뮤트",
    "Siberian_husky": "시베리안 허스키",
    "Affenpinscher": "아펜핀셔",
    "Basenji": "바센지",
    "Pug": "퍼그",
    "Leonberg": "레온베르거",
    "Newfoundland": "뉴펀들랜드",
    "Great_Pyrenees": "그레이트 피레니즈",
    "Samoyed": "사모예드",
    "Pomeranian": "포메라니안",
    "Chow": "차우차우",
    "Keeshond": "키스혼트",
    "Brabancon_griffon": "브라반손 그리펀",
    "Pembroke": "펨브로크 웰시 코기",
    "Cardigan": "카디건 웰시 코기",
    "Toy_poodle": "토이 푸들",
    "Miniature_poodle": "미니어처 푸들",
    "Standard_poodle": "스탠다드 푸들",
    "Mexican_hairless": "멕시칸 헤어리스",
    "Dingo": "딩고",
    "Dhole": "도울",
    "African_hunting_dog": "아프리칸 헌팅 독"
}

chat_analysis_template = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 20년 경력의 수의사이자 강아지 행동 분석 전문가입니다. 주어진 모든 정보를 종합하여 강아지의 상태를 분석하고 전문적인 조언을 제공해야 합니다.

    ** 입력 정보 ** 
    1. YOLO 분석 결과: {yolo_text}
    2. LSTM 분석 결과: {lstm_text}
    3. LLM 비디오 분석 결과: {summary_text}
    4. RAG 문서 정보: {rag_text}

    ** 유저와의 상호작용 ** 
    모든 대화는 한국어로 진행됩니다.

    ** 분석 및 답변 지침 ** 
    1. 모든 입력 정보를 종합하여 강아지의 상태, 행동, 감정, 건강 상태를 정확히 파악하세요.
    2. YOLO 결과로부터 강아지의 품종과 외형적 특징을 고려하세요.
    3. LSTM 결과를 바탕으로 강아지의 행동 패턴과 감정 상태, 통증여부, 이상행동 여부를 분석하세요.
    4. LLM 비디오 분석 결과를 통해 전반적인 상황 맥락을 이해하세요.
    5. RAG 문서 정보를 활용하여 관련된 전문 지식을 답변에 통합하세요.
    6. 문제가 있다면 그 원인을 간단히 설명하고, 구체적이고 실행 가능한 해결책을 제시하세요.
    7. 보호자가 즉시 실천할 수 있는 실용적인 조언을 포함하세요.
    8. 필요한 경우 전문가 상담이나 병원 방문을 권유하세요.
    9. 답변은 전문적이면서도 이해하기 쉽게 작성하세요.
    
    ** 답변 구조 ** 
    유저의 질문에 대해 주어진 모든 정보를 바탕으로 한 문단으로 종합적이고 직접적인 답변을 작성하세요.

    주어진 모든 정보를 종합하여 전문적이고 실용적인 분석과 조언을 제공해주세요.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

comprehensive_analysis_template = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 20년 경력의 수의사이자 강아지 행동 분석 전문가입니다. 주어진 모든 정보를 종합하여 강아지의 상태를 분석하고 전문적인 조언을 제공해야 합니다.

    ** 입력 정보 **
    1. YOLO 분석 결과: {yolo_text}
    2. LSTM 분석 결과: {lstm_text}
    3. LLM 비디오 분석 결과: {summary_text}
    4. RAG 문서 정보: {rag_text}

    ** 유저와의 상호작용 **
    모든 대화는 한국어로 진행됩니다.

    ** 분석 및 답변 지침 **
    1. 모든 입력 정보를 종합하여 강아지의 상태, 행동, 감정, 건강 상태를 정확히 파악하세요.
    2. YOLO 결과로부터 강아지의 품종과 외형적 특징을 고려하세요.
    3. LSTM 결과를 바탕으로 강아지의 행동 패턴과 감정 상태, 통증여부, 이상행동 여부를 분석하세요.
    4. LLM 비디오 분석 결과를 통해 전반적인 상황 맥락을 이해하세요.
    5. RAG 문서 정보를 활용하여 관련된 전문 지식을 답변에 통합하세요.
    6. 문제가 있다면 그 원인을 간단히 설명하고, 구체적이고 실행 가능한 해결책을 제시하세요.
    7. 보호자가 즉시 실천할 수 있는 실용적인 조언을 포함하세요.
    8. 필요한 경우 전문가 상담이나 병원 방문을 권유하세요.
    9. 답변은 전문적이면서도 이해하기 쉽게 작성하세요.
    
    ** 답변 구조 **
    1. 종합적 상황 요약 (YOLO, LSTM, LLM 결과 통합)
    2. 원인 분석 (해당되는 경우)
    3. 맞춤형 해결책 또는 권장 행동 (RAG 정보 활용)
    4. 추가 조언 또는 주의사항 (필요한 경우)
    5. 결론 및 격려의 말

    주어진 모든 정보를 종합하여 전문적이고 실용적인 분석과 조언을 제공해주세요.
    """),
    MessagesPlaceholder(variable_name="chat_history")
])

def process_yolo_results(results):
    lstm_keypoint_sequence = []
    skeleton_sequence = []
    breed_counter = Counter()

    for r in results:
        if r.keypoints is not None and len(r.keypoints) > 0:
            yolo_keypoints = r.keypoints[0].cpu().numpy()
            lstm_keypoints = convert_yolo_to_lstm(yolo_keypoints)
            lstm_keypoint_sequence.append(lstm_keypoints)
            skeleton = create_skeleton(lstm_keypoints)
            skeleton_sequence.append(skeleton)

            # YOLO 결과에서 품종 정보 추출 및 카운트
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = r.names[cls]
                breed_counter[class_name] += 1  # 품종 등장 횟수 증가

    return np.array(lstm_keypoint_sequence), np.array(skeleton_sequence), breed_counter

def convert_yolo_to_lstm(yolo_keypoints):
    image_width, image_height = 640, 384
    yolo_to_lstm_mapping = {
        16: 0, 23: 4, 8: 5, 2: 6, 7: 7, 1: 8, 10: 9, 4: 10, 9: 11, 3: 12, 12: 13, 13: 14
    }
    lstm_keypoints = np.zeros((15, 2), dtype=float)
    
    for yolo_index, lstm_index in yolo_to_lstm_mapping.items():
        if yolo_index < yolo_keypoints.shape[0]:
            x, y = yolo_keypoints[yolo_index]
            lstm_keypoints[lstm_index] = [x / image_width, y / image_height]
    
    # 이마, 입꼬리, 아래 입술 중앙 처리 (가능한 경우에만)
    if 20 < yolo_keypoints.shape[0]:
        lstm_keypoints[1] = yolo_keypoints[20] / np.array([image_width, image_height])
    if 17 < yolo_keypoints.shape[0]:
        lstm_keypoints[2] = lstm_keypoints[3] = yolo_keypoints[17] / np.array([image_width, image_height])
    
    return lstm_keypoints

def create_skeleton(keypoints):
    DOG_SKELETON = [
        [0, 1], [0, 2], [2, 3], [1, 4], [4, 5], [4, 6], [5, 7], [6, 8],
        [9, 11], [10, 12], [4, 13], [13, 14], [9, 13], [10, 13], [5, 9],
        [6, 10], [5, 6], [9, 10]
    ]
    skeleton = []
    for start, end in DOG_SKELETON:
        start_point, end_point = keypoints[start], keypoints[end]
        skeleton.extend([start_point[0], start_point[1], end_point[0], end_point[1]])
    return skeleton

# YOLO 결과 처리 및 LSTM 입력 준비
def process_video(video_path, lstm_text, yolo_text):
    if not os.path.exists(video_path):
        print(f"오류: 비디오 파일을 찾을 수 없습니다: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 비디오 파일을 열 수 없습니다: {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = 'output_video_with_labels.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 한글 폰트 설정
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)

    try:
        for _ in tqdm(range(total_frames), desc="비디오 처리 중"):
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽는 데 실패했습니다.")
                break

            # OpenCV 이미지를 Pillow 이미지로 변환
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            # YOLO 텍스트 추가
            y = 30
            for line in yolo_text.split(', '):
                draw.text((10, y), line, font=font, fill=(255, 255, 255))
                y += 30

            # LSTM 텍스트 추가
            y += 30
            for line in lstm_text.split(', '):
                draw.text((10, y), line, font=font, fill=(255, 255, 255))
                y += 30

            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            out.write(frame)

    except Exception as e:
        print(f"비디오 처리 중 오류 발생: {str(e)}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    if os.path.exists(output_path):
        print(f"레이블이 추가된 비디오가 {output_path}에 저장되었습니다.")
        return output_path
    else:
        print("오류: 출력 비디오 파일이 생성되지 않았습니다.")
        return None


# LLM 분석 템플릿
llm_analysis_template = """
    ** 역할 **
    당신은 20년간 강아지에 대해 공부한 수의사 및 행동 분석가입니다.
    강아지에 대한 풍부한 경험과 전문적인 지식을 보유하고 있습니다.
    제공된 영상과 오디오를 확인하여 강아지에 대한 분석을 진행합니다.

    ** 분석 프로세스 **
    1. 영상 분석:
       - 제공된 프레임 단위 이미지를 순서대로 분석합니다.
       - 강아지의 자세, 움직임, 표정을 관찰합니다.
       - 주변 환경과 상황적 맥락을 파악합니다.
       - 보호자가 있다면 그들의 행동도 분석합니다.

    2. 오디오 분석:
       - 오디오 파형과 주파수 분석으로 강아지의 소리(짖음, 울음 등)의 특성을 파악합니다.
       - 배경 소음이나 다른 소리들도 고려합니다.

    3. 종합 분석:
       - 영상과 오디오 정보를 종합하여 강아지의 전반적인 상태를 평가합니다.
       - 행동 패턴, 감정 상태, 건강 상태에 대한 의견을 제시합니다.
       - 특이사항이나 문제점이 있다면 언급합니다.

    ** 분석 결과 형식 **
    분석 결과를 다음 형식으로 제공해 주세요:
    1. 영상 분석 요약: [영상에서 관찰된 주요 사항들을 간결하게 서술]
    2. 오디오 분석 요약: [오디오에서 파악된 주요 정보를 간결하게 서술]
    3. 종합 분석: [영상과 오디오 정보를 종합한 전반적인 분석 결과]

    주어진 정보를 바탕으로 강아지의 상태를 전문가적 관점에서 분석해 주세요.
"""


async def analyze_video(file_path, seconds_per_frame=2):
    base64Frames = []
    base_video_file, _ = os.path.splitext(file_path)
    video = cv2.VideoCapture(file_path)
    if not video.isOpened():
        raise Exception("Error opening video file")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)

    curr_frame = 0
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break

        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip

    video.release()

    clip = VideoFileClip(file_path)
    audio_path = f"{base_video_file}.mp3"
    try:
        if clip.audio:
            await asyncio.to_thread(clip.audio.write_audiofile, audio_path, bitrate="32k")
            clip.audio.close()
        else:
            audio_path = None
    except Exception as e:
        audio_path = None
    clip.close()

    return base64Frames, audio_path

# Define function to summarize video
async def summarize_video(base64Frames, audio_path):
    summary_text = ""

    # 비디오 프레임을 표시합니다.
    if base64Frames:  # base64Frames가 비어있지 않은지 확인
        for img in base64Frames:
            # base64로 인코딩된 이미지를 디코딩하여 PIL 이미지로 변환
            image_data = base64.b64decode(img)
            image = Image.open(io.BytesIO(image_data))  # BytesIO를 사용하여 이미지 열기
            
    # 오디오 경로가 None이 아닌 경우에만 transcription을 실행합니다.
    if audio_path is not None:
        with open(audio_path, 'rb') as audio_file:
            transcription = await asyncio.to_thread(client.audio.transcriptions.create,
                                                    model="whisper-1",
                                                    file=audio_file)
            audio_text = transcription.text

            print("Transcription 완료:", audio_text, '\n')
            summary_text += audio_text + "\n"

            response_both = await asyncio.to_thread(client.chat.completions.create,
                                                    model=model,
                                                    messages=[
                                                        {"role": "system", "content": llm_analysis_template},
                                                        {"role": "user", "content": [
                                                            "이건 비디오 영상의 프레임 이미지.",
                                                            *map(lambda x: {"type": "image_url",
                                                                            "image_url": {"url": f"data:image/jpg;base64,{x}", "detail":"low"}}, base64Frames),
                                                            {"type": "text", "text": f"이건 비디오 영상의 오디오 {audio_text}"}
                                                        ]},
                                                    ],
                                                    temperature=0.6)
            print(response_both.choices[0].message.content)
            summary_text += response_both.choices[0].message.content + "\n"
            print("\n", "="*100, "\n")

    else:
        print("오디오 내용이 없습니다. Transcription은 스킵합니다", '\n')

        # 오디오가 없는 경우에도 비디오 프레임에 대한 분석을 수행합니다.
        response_vis = await asyncio.to_thread(client.chat.completions.create,
                                               model=model,
                                               messages=[
                                                   {"role": "system", "content": llm_analysis_template},
                                                   {"role": "user", "content": [
                                                       "이건 비디오 영상의 프레임 이미지.",
                                                       *map(lambda x: {"type": "image_url",
                                                                       "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames)
                                                   ]},
                                               ],
                                               temperature=0.6)
        print(response_vis.choices[0].message.content)
        summary_text += response_vis.choices[0].message.content + "\n"
        print("\n", "="*100, "\n")

    return summary_text

def generate_query(yolo_text, lstm_text, summary_text):
    # 쿼리 템플릿에 통합된 텍스트를 삽입
    query_template = f"""
    당신은 강아지 행동 분석 전문가입니다. 주어진 정보를 바탕으로 강아지의 상태를 종합적으로 요약해야 합니다.

    ** 입력 정보 **
    1. YOLO 모델 결과: {yolo_text} (강아지 품종 정보)
    2. LSTM 모델 결과: {lstm_text} (강아지 행동, 감정, 통증여부, 질병여부, 비정상 행동 정보)
    3. LLM 비디오/오디오 분석 요약: {summary_text} (강아지의 시각적 행동 패턴, 소리, 환경 정보 등)

    ** 요약 지침 **
    1. 모든 입력 정보를 통합하여 강아지의 상태를 종합적으로 설명하세요.
    2. 품종, 주요행동, 감정상태, 건강상태, 환경적 요인을 포함해주세요.
    3. 영상 속 강아지의 특이사항이나 문제점, 솔루션이 필요하다면 말해주세요.
    4. 전체 요약은 2-3문장으로 제한하세요.
    5. 요약은 키워드 중심으로 작성하고, 불필요한 관사나 조사는 생략하세요.

    위의 지침을 참고하여, 주어진 모든 정보를 바탕으로 강아지의 상태를 종합적으로 요약해주세요.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": query_template},
            {"role": "user", "content": "비디오를 종합적으로 요약해주세요."}
        ],
        max_tokens=100
    )

    query = response.choices[0].message.content
    return query

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
    
    # 저장된 TfidfVectorizer 로드
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    
    # 커스텀 임베딩 클래스 정의
    class CustomEmbeddings:
        def __init__(self, embeddings_model, vectorizer):
            self.embeddings_model = embeddings_model
            self.vectorizer = vectorizer
        
        def embed_query(self, text):
            dense_embedding = self.embeddings_model.embed_query(text)
            sparse_embedding = self.vectorizer.transform([text]).toarray()[0]
            combined_embedding = np.concatenate([dense_embedding, sparse_embedding])
            if combined_embedding.shape[0] > 2048:
                combined_embedding = combined_embedding[:2048]
            elif combined_embedding.shape[0] < 2048:
                padding = np.zeros(2048 - combined_embedding.shape[0])
                combined_embedding = np.concatenate([combined_embedding, padding])
            return combined_embedding.tolist()
    
    custom_embeddings = CustomEmbeddings(embeddings_model, vectorizer)
    
    # MongoDB 컬렉션 설정
    dbName_json = "db"
    collectionName_json = "dog_document"
    collection_json = db_client[dbName_json][collectionName_json]

    vectorStore_json = MongoDBAtlasVectorSearch(
        embedding=custom_embeddings,
        collection=collection_json,
        index_name='dog_index2',
        embedding_key="embedding",
        text_key="content"
    )
    # 검색기 생성
    retriever_json = vectorStore_json.as_retriever()
    return retriever_json

def chat_analysis(yolo_text, lstm_text, summary_text, rag_text, user_question, chat_history):
    try:
        # Creating the prompt message
        prompt_message = chat_analysis_template.format(
            yolo_text=yolo_text,
            lstm_text=lstm_text,
            summary_text=summary_text,
            rag_text=rag_text,
            question=user_question,
            chat_history=chat_history
        )
        
        # Create the messages list
        messages = [
            {"role": "system", "content": prompt_message},
            {"role": "user", "content": user_question}
        ]
        
        # Request completion from the API
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18", 
            messages=messages,
            max_tokens=1000,
            temperature=0.6
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"generate_comprehensive_analysis 중 오류 발생: {str(e)}")
        return ""

# 초기 분석 결과
def generate_comprehensive_analysis(yolo_text, lstm_text, summary_text, rag_text, chat_history):
    try: 
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18", 
            messages=[
                {"role": "system", "content": comprehensive_analysis_template.format(
                    yolo_text=yolo_text,
                    lstm_text=lstm_text,
                    summary_text=summary_text,
                    rag_text=rag_text,
                    # question=question,
                    chat_history=chat_history
                )},
                {"role": "user", "content": "강아지의 상태를 분석하고 조언해주세요."}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"generate_comprehensive_analysis 중 오류 발생: {str(e)}")
        return ""

# Define function to get most common breed
def get_most_common_breed(breed_counter):
    if breed_counter:
        most_common_breed = breed_counter.most_common(1)[0][0]
        total_detections = sum(breed_counter.values())
        breed_percentage = (breed_counter[most_common_breed] / total_detections) * 100
        return most_common_breed, breed_percentage
    return "믹스", 0

# Define async function to analyze video and generate text
async def analyze_video_and_generate_text(video_path, yolo_model, device, conf_thresh, checkpoint, font_path):
    # YOLO 모델 예측
    results = await asyncio.to_thread(yolo_model.predict, source=video_path, save=True, conf=conf_thresh, stream=True, verbose=False)
    lstm_keypoint_sequence, skeleton_sequence, breed_counter = process_yolo_results(results)

    # 가장 많이 탐지된 품종 선택
    most_common_breed, breed_percentage = get_most_common_breed(breed_counter)
    most_common_breed_ko = breed_mapping.get(most_common_breed, most_common_breed)
    yolo_result = f"가장 많이 탐지된 품종: {most_common_breed_ko} (전체 탐지 중 {breed_percentage:.2f}%)"
    yolo_text = most_common_breed_ko

    print("YOLO 분석 결과:")
    print(yolo_result)
    print(f"yolo_text: {yolo_text}")

    # LSTM 분석
    class ImprovedLSTMModel(nn.Module):
        def __init__(self, keypoint_size, skeleton_size, hidden_size, num_layers, num_classes):
            super(ImprovedLSTMModel, self).__init__()
            self.keypoint_lstm = nn.LSTM(keypoint_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
            self.skeleton_lstm = nn.LSTM(skeleton_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
            self.fc = nn.Linear(hidden_size * 2, num_classes)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, keypoints, skeleton):
            _, (h_n_keypoints, _) = self.keypoint_lstm(keypoints)
            _, (h_n_skeleton, _) = self.skeleton_lstm(skeleton)
            combined = torch.cat((h_n_keypoints[-1], h_n_skeleton[-1]), dim=1)
            out = self.dropout(combined)
            out = self.fc(out)
            return out

    # 모델 파라미터 및 메타데이터 설정
    keypoint_size = checkpoint.get('keypoint_size', 30)
    skeleton_size = checkpoint.get('skeleton_size', 72)
    hidden_size = checkpoint.get('hidden_size', 128)
    num_layers = checkpoint.get('num_layers', 2)
    num_classes = checkpoint.get('num_classes', len(checkpoint.get('all_class_names', [])))
    all_class_names = checkpoint.get('all_class_names', [f'Class_{i}' for i in range(num_classes)])
    metadata = checkpoint.get('metadata', []) 

    # 모델 인스턴스 생성
    lstm_model = ImprovedLSTMModel(keypoint_size, skeleton_size, hidden_size, num_layers, num_classes).to(device)

    # 가중치 직접 로드
    try:
        lstm_model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        print("'model_state_dict' 키를 찾을 수 없습니다. 체크포인트에서 직접 가중치를 로드합니다.")
        model_dict = lstm_model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        lstm_model.load_state_dict(model_dict)

    lstm_model.eval()

    # 슬라이딩 윈도우 함수 정의
    def dynamic_sliding_window(sequence, min_window_size=50, max_window_size=100, min_stride=10):
        sequence_length = len(sequence)
        window_size = min(max_window_size, max(min_window_size, sequence_length // 10))
        stride = max(min_stride, window_size // 2)
        
        windows = []
        for start in range(0, sequence_length - window_size + 1, stride):
            windows.append(sequence[start:start + window_size])
        
        if sequence_length % window_size != 0:
            windows.append(sequence[-window_size:])
        
        return np.array(windows)

    # 슬라이딩 윈도우 적용
    lstm_windows = dynamic_sliding_window(lstm_keypoint_sequence)
    skeleton_windows = dynamic_sliding_window(skeleton_sequence)

    print(f"LSTM 윈도우 형태: {lstm_windows.shape}")
    print(f"스켈레톤 윈도우 형태: {skeleton_windows.shape}")

    # 텐서로 변환
    keypoints_tensor = torch.FloatTensor(lstm_windows).to(device)
    skeleton_tensor = torch.FloatTensor(skeleton_windows).to(device)

    print(f"키포인트 텐서 형태: {keypoints_tensor.shape}")
    print(f"스켈레톤 텐서 형태: {skeleton_tensor.shape}")

    # 키포인트 텐서 형태 조정 (필요한 경우)
    if keypoints_tensor.shape[-2] == 15 and keypoints_tensor.shape[-1] == 2:
        keypoints_tensor = keypoints_tensor.view(keypoints_tensor.shape[0], keypoints_tensor.shape[1], -1)

    print(f"조정된 키포인트 텐서 형태: {keypoints_tensor.shape}")
    print(f"조정된 스켈레톤 텐서 형태: {skeleton_tensor.shape}")

    # LSTM 입력 형태 확인
    assert keypoints_tensor.dim() == 3, f"키포인트 텐서는 3차원이어야 합니다. 현재: {keypoints_tensor.dim()}D"
    assert skeleton_tensor.dim() == 3, f"스켈레톤 텐서는 3차원이어야 합니다. 현재: {skeleton_tensor.dim()}D"
    assert keypoints_tensor.shape[-1] == 30, f"키포인트 텐서의 마지막 차원은 30이어야 합니다. 현재: {keypoints_tensor.shape[-1]}"
    assert skeleton_tensor.shape[-1] == 72, f"스켈레톤 텐서의 마지막 차원은 72여야 합니다. 현재: {skeleton_tensor.shape[-1]}"

    # 예측
    with torch.no_grad():
        outputs = lstm_model(keypoints_tensor, skeleton_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

    # 가장 많이 예측된 클래스 선택
    predicted_class = predicted_classes.mode().values.item()
    overall_probabilities = probabilities.mean(dim=0)

    # 예측 결과와 연관된 메타데이터 가져오기
    predicted_action = all_class_names[predicted_class]
    associated_metadata = metadata[predicted_class] if predicted_class < len(metadata) else {}

    # 예측 결과 출력
    print(f"예측된 클래스: {predicted_action}")
    print(f"클래스별 확률:")
    for i, prob in enumerate(overall_probabilities):
        print(f"  {all_class_names[i]}: {prob.item():.4f}")

    # 메타데이터 출력
    print("\n예측된 클래스의 메타데이터:")
    if associated_metadata:
        print(f"  통증: {associated_metadata.get('pain', 'N/A')}")
        print(f"  질병: {associated_metadata.get('disease', 'N/A')}")
        print(f"  감정: {associated_metadata.get('emotion', 'N/A')}")
        print(f"  비정상 행동: {associated_metadata.get('abnormal_action', 'N/A')}")
    else:
        print("  이 클래스에 대한 메타데이터가 없습니다.")

    # lstm_text 생성
    lstm_text = f"행동: {predicted_action}\n"
    if associated_metadata:
        lstm_text += f"  통증: {associated_metadata.get('pain', 'N/A')}\n"
        lstm_text += f"  질병: {associated_metadata.get('disease', 'N/A')}\n"
        lstm_text += f"  감정: {associated_metadata.get('emotion', 'N/A')}\n"
        lstm_text += f"  비정상 행동: {associated_metadata.get('abnormal_action', 'N/A')}\n"
    else:
        lstm_text += "알 수 없음"

    # LLM 분석
    base64Frames, audio_path = await analyze_video(video_path)
    summary_text = await summarize_video(base64Frames, audio_path)

    # Query 생성
    summary_query = generate_query(yolo_text, lstm_text, summary_text)
    print("query :", summary_query)

    # 벡터 검색
    retriever = get_ret()
    try:
        documents = await asyncio.to_thread(retriever.invoke, summary_query)

        print("검색된 문서:")
        if not documents:
            print("검색 결과가 없습니다.")
            rag_text = "검색 결과가 없습니다."
        else:
            # RAG 텍스트로 변환
            rag_text = "\n\n".join([doc.page_content for doc in documents])
            print("RAG Text:")
            print(rag_text)
    except Exception as e:
        print(f"검색 중 오류가 발생했습니다: {str(e)}")
        rag_text = "검색 중 오류가 발생했습니다."

    # 종합 분석 생성
    follow_up_answer = generate_comprehensive_analysis(
        yolo_text=yolo_text,
        lstm_text=lstm_text,
        summary_text=summary_text,
        rag_text=rag_text,
        chat_history=[]  # 초기 분석 결과 포함
    )

    print("Follow Up Answer:")
    print(follow_up_answer)

    # 비디오 처리 및 라벨 추가 (선택 사항)
    # labeled_video_path = await asyncio.to_thread(process_video, video_path, lstm_text, yolo_text)

    # 결과 반환
    return {
        "yolo_text": yolo_text,
        "lstm_text": lstm_text,
        "summary_text": summary_text,
        "rag_text": rag_text,
        "follow_up_answer": follow_up_answer
    }

# 추가 상호작용 기존에 있던거
# async def handle_user_question(session_id: str, user_question: str, chat_history: List[Dict[str, Any]], analysis_results: Dict[str, Any]):

#     try:
#         # 분석 결과 추출
#         yolo_text = analysis_results.get("yolo_text", "")
#         lstm_text = analysis_results.get("lstm_text", "")
#         summary_text = analysis_results.get("summary_text", "")
#         rag_text = analysis_results.get("rag_text", "")

#         # 새로운 질문으로 래그 
#         # 벡터 검색
#         retriever = get_ret()
#         try:
#             documents = retriever.invoke(user_question)

#             logger.info("검색된 문서:")
#             if not documents:
#                 logger.info("검색 결과가 없습니다.")
#                 rag_text = "검색 결과가 없습니다."
#             else:
#                 # RAG 텍스트로 변환
#                 rag_text = "\n\n".join([doc.page_content for doc in documents])
#                 logger.info("RAG Text:")
#                 logger.info(rag_text)
#         except Exception as e:
#             logger.error(f"검색 중 오류가 발생했습니다: {str(e)}")
#             rag_text = "검색 중 오류가 발생했습니다."

#         # 사용자 질문을 chat_history에 추가
#         chat_history.append({"role": "user", "content": user_question})

#         # 분석 결과 생성
#         follow_up_answer = chat_analysis(
#             yolo_text=yolo_text,
#             lstm_text=lstm_text,
#             summary_text=summary_text,
#             rag_text=rag_text,
#             user_question=user_question,  # 사용자 질문으로 설정
#             chat_history=chat_history  # 대화 기록으로 설정
#         )

#         # 대화 내역에 응답 추가
#         chat_history.append({"role": "assistant", "content": follow_up_answer})

#         # 결과 반환
#         # print(f'종합 답변 : {follow_up_answer}')
#         # return follow_up_answer
#         # 결과 반환 
#         # print(f'종합 답변 : {follow_up_answer}')
#         return {
#             'answer': follow_up_answer,
#         }
#     except Exception as e:
#         logger.error(f"handle_user_question 중 오류 발생: {str(e)}")
#         return "죄송합니다. 질문을 처리하는 중 오류가 발생했습니다."


# 새로 만든거
async def handle_user_question(user_question: str, chat_history: dict, analysis_results: dict):
    try:
        # MongoDB에서 대화 기록 불러오기
        # chat_history = await get_chat_history(user_email)

        # 분석 결과 추출 (이전 분석 결과가 필요할 경우)
        yolo_text = analysis_results.get("yolo_text", "")
        lstm_text = analysis_results.get("lstm_text", "")
        summary_text = analysis_results.get("summary_text", "")
        rag_text = analysis_results.get("rag_text", "")

        # 새로운 질문으로 래그 
        # 벡터 검색
        retriever = get_ret()
        try:
            documents = retriever.invoke(user_question)

            logger.info("검색된 문서:")
            if not documents:
                logger.info("검색 결과가 없습니다.")
                rag_text = "검색 결과가 없습니다."
            else:
                # RAG 텍스트로 변환
                rag_text = "\n\n".join([doc.page_content for doc in documents])
                logger.info("RAG Text:")
                logger.info(rag_text)
        except Exception as e:
            logger.error(f"검색 중 오류가 발생했습니다: {str(e)}")
            rag_text = "검색 중 오류가 발생했습니다."


        # 사용자 질문을 chat_history에 추가
        # chat_history.append({"role": "user", "content": user_question})

        # 분석 결과 생성
        follow_up_answer = chat_analysis(
            yolo_text=yolo_text,
            lstm_text=lstm_text,
            summary_text=summary_text,
            rag_text=rag_text,
            user_question=user_question,
            chat_history=chat_history
        )

        # 대화 내역에 응답 추가
        # chat_history.append({"role": "assistant", "content": follow_up_answer})

        # MongoDB에 대화 기록 업데이트
        # await update_chat_history(user_email, user_question, follow_up_answer)

        return {
            'answer': follow_up_answer,
        }
    except Exception as e:
        logger.error(f"handle_user_question 중 오류 발생: {str(e)}")
        return "죄송합니다. 질문을 처리하는 중 오류가 발생했습니다."
