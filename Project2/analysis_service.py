# analysis_service.py
import os
import base64
import cv2
from conversation import ConversationManager, chat_history_collection  # 이 부분을 추가
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from moviepy.editor import VideoFileClip
from pymongo import MongoClient
from openai import OpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate

from moviepy.editor import VideoFileClip
from openai import OpenAI
from pymongo import MongoClient
from retriever import get_ret
import tiktoken
import json

# MongoDB 연결 설정
url = f"mongodb+srv://yoonsun2596:qwer1234@tm2.hl7a3.mongodb.net/"
db_client = MongoClient(url)
db = db_client['Test']
video_analysis_collection = db['video_analysis']
chat_history_collection = db['chat_log']

# OpenAI API 키 설정
client = OpenAI(api_key="API 키 필요")


system_msg = """
** 역할 **
"당신은 20년 이상의 경력을 가진 숙련된 변호사이다. 도로교통법 분야에서 풍부한 경험과 전문 지식을 보유하고 있습니다. 당신의 역할은 교통사고에 관한 질문내용에 대해 전문적이고 실용적인 법률 조언을 제공하는 것입니다. 상황설명을 고려하여 해당 사례에 과실비율, 적용 가능한 법률, 판례, 그리고 실제 사례를 참조하여 상세한 답변을 제공해 주세요."
** 유저와의 상호작용 **
모든 대화는 한국어로 진행됩니다.
** 답변시 다음 사항을 반드시 포함해 주세요:
1. 해당 사고 상황에 대한 과실비율 (몇대몇).
2. 사고 상황에서의 주요 쟁점.
3. 결정 이유와 근거
4. 사고상황, 사고내용에 적합한 애니메이션 'url' 제공한다.
5. 관련 법률 조항 및 해석.
6. 구체적인 해결 방안 또는 법적 절차 안내.
7. 주의해야 할 법적 리스크나 고려 사항.
8. 위 내용을 요약.정리한 결론.
"""

video_msg = """
** 역할 설명 **
"당신은 20년 이상의 경력을 가진 숙련된 블랙박스 영상 분석가입니다. 당신의 역할은 교통사고 현장에서 촬영된 블랙박스 영상을 분석하여 사고의 원인, 과실 비율, 그리고 법률적 판단에 도움을 되는 정보를 제공하는 것 입니다."
** 서비스개요 **
서비스 개요 : "블랙박스 영상 파일을 분석하여 교통사고 상황을 요약하고 분석하여 사용자에게 제공한다."
** 유저와의 상호작용 **
모든 대화는 한국어로 진행됩니다.
** 답변 시 다음 사항을 반드시 포함해 주세요.
1. 사고 상황 : 운전자의 입점에서 사고영상을 처음부터 끝까지 스토리 형식으로 작성해 주세요.
2. 사고 분석 : 운전자의 입장에서 영상 분석하여 사고 발생한 외부 요인을 정확하고 자세하게 알려주세요.
3. 주요 쟁점 : 사고 분석 내용에 근거하여 각 당사자들간의 잘못과 사건의 핵심 쟁점을 알려주세요.
4. 과실 비율 : 도로교통법에 근거하여 과실 비율을 판단해주세요.
"""
prompt = ChatPromptTemplate.from_messages(
    messages=[
        ('system',
            """
            ** 역할 **
            "당신은 20년 이상의 경력을 가진 숙련된 교통사고 전문 변호사이다. 도로교통법 분야에서 풍부한 경험과 전문 지식을 보유하고 있습니다. 당신의 역할은 교통사고에 관한 질문내용에 대해 전문적이고 실용적인 법률 조언을 제공하는 것입니다. 사용자의 상황을 고려하여 해당 사고에 대한 과실비율, 관련법률, 판례에 대한 정보를 문서에서 찾아서 상세한 답변을 제공해 주세요."
            ** 유저와의 상호작용 **
            모든 대화는 한국어로 진행됩니다.
            ** 유저의 질문에서 데이터가 부족하면 MAX 1번만 다시 되물어 교통사고에 대한 데이터를 수집합니다 안 부족하면 바로 답변합니다
            - 제발 1번만 유저에게 물어봐
            - 물어 볼 질문에는 교통사고 났을 시 상황, 자동차의 피해, 신호등의 유무/색깔, 사고난 장소를 중점적으로 물어봅니다
            - 유저에게 질문 1개 이후 더 이상 질문은 하지 않고 받은 정보만으로 답변을 합니다
            - 질문이 더 필요해도 꼭 1개만 물어보고 가진 정보로 대답합니다
            - 유저가 비슷한 케이스/사고를 물어보면 지금까지 유저가 물어본 교통사고와 비슷한 케이스를 찾고 그 케이스의 사고 상황과 과실비율을 유저에게 알려줘
            - 유저가 비슷한 케이스/사고를 물어보면 절대 되묻지 말고 찾은 데이터에서 유저와 비슷한 사고의 답변을 하단과 똑같은 패턴으로 알려줘


            ** 답변시 다음 사항을 반드시 포함해 주세요:
            1. 사고 상황에 대한 과실비율. (general_json.liabilityRatio, if_json.liabilityRatio 참조)
            2. 사고 상황에서의 주요 쟁점. (general_json.animationURL, if_json.accidentDetails 참조)
            3. 결정 이유와 근거.  (general_json.liabilityExplanation, if_json.decisionReason 참조)
            4. 관련 법률 조항 및 해석. (general_json.relatedLaws)
            5. 구체적인 해결 방안 또는 법적 절차 안내. (general_json.judicialCases 참조)
            6. 주의해야 할 법적 리스크나 고려 사항.
            7. 위 내용을 요약.정리한 결론.

            순서대로 생각해서 대답해주세요.

            마지막에 해당 답변에 만족하는지 아닌지 물어본다
            
            \n\n
            Context: {context}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
    ]
)

# Prompt templates
system_template = ChatPromptTemplate.from_messages(
    messages=[
        ('system',
            """
            role : You are an experienced lawyer with over 20 years of experience in traffic law. 
            Communication method: Please communicate with users in Korean.
            Provide professional and practical legal advice on traffic accident inquiries. 
            Consider the situation description and provide detailed answers, including:
            1. Fault ratio for the accident.
            2. Main issues in the accident situation.
            3. Reasons and basis for the decision.
            4. Relevant legal clauses and interpretation.
            5. Specific resolution or legal procedure guidance.
            6. Legal risks or considerations to be aware of.
            7. Summarized conclusion of the above content.
            Please think in order and answer.

            \n\n
            Context: {context}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
    ]
)

# Video analysis chain
video_template = ChatPromptTemplate.from_messages(
    messages=[
        ('system',
            """
Analyze the following video content:
Frames: {frames}
Audio transcript: {audio_transcript}

Provide a detailed analysis of the traffic accident shown in the video, including:
1. Accident situation: Describe the accident from the driver's perspective, from start to finish.
2. Accident analysis: Analyze the external factors that caused the accident from the driver's perspective.
3. Main issues: Based on the analysis, identify the faults of each party and the core issues of the incident.
4. Fault ratio: Determine the fault ratio based on traffic laws.
\n\n
            Context: {context}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
    ]
)

functions = [
    {
        "name": "analyze_text",
        "description": "사용자가 입력한 텍스트에 대한 답변을 제공합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "분석할 텍스트"
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "analyze_video_with_interaction",
        "description": "비디오 분석과 사용자의 질문에 대한 답변을 제공합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "비디오 파일의 경로"
                }
            },
            "required": ["file_path"]
        }
    }
]

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """주어진 문자열의 토큰 수를 반환합니다."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def optimize_context(documents, max_tokens=3000):
    """컨텍스트를 최적화하여 토큰 수를 제한합니다."""
    optimized_context = ""
    current_tokens = 0
    
    for doc in documents:
        doc_tokens = num_tokens_from_string(doc, "cl100k_base")
        if current_tokens + doc_tokens > max_tokens:
            break
        optimized_context += doc + "\n\n"
        current_tokens += doc_tokens
    
    return optimized_context.strip()

def post_process_response(response, animation_url=None):
    """LLM의 응답을 후처리하여 가독성을 높입니다."""
    # 응답에서 각 섹션을 추출
    sections = {
        "과실비율": "",
        "주요 쟁점": "",
        "결정 이유와 근거": "",
        "애니메이션 URL": animation_url if animation_url else "",  # 애니메이션 URL 포함
        "관련 법률 조항 및 해석": "",
        "해결 방안 또는 법적 절차": "",
        "법적 리스크 및 고려 사항": "",
        "결론": ""
    }
    
    current_section = None
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue  # 빈 줄은 무시
        
        for section in sections:
            if section in line:
                current_section = section
                sections[current_section] = ""  # 초기화
                break
        if current_section:
            sections[current_section] += line + '\n'
    
    # 구조화된 응답 생성
    structured_response = {
        "과실비율": sections["과실비율"].strip(),
        "주요_쟁점": sections["주요 쟁점"].strip(),
        "결정_이유": sections["결정 이유와 근거"].strip(),
        "애니메이션_URL": sections["애니메이션 URL"].strip(),
        "관련_법률": sections["관련 법률 조항 및 해석"].strip(),
        "해결_방안": sections["해결 방안 또는 법적 절차"].strip(),
        "법적_리스크": sections["법적 리스크 및 고려 사항"].strip(),
        "결론": sections["결론"].strip()
    }

    formatted_response = (
        f"**과실비율**\n{structured_response['과실비율']}\n\n"
        f"**주요 쟁점**\n{structured_response['주요_쟁점']}\n\n"
        f"**결정 이유 및 근거**\n{structured_response['결정_이유']}\n\n"
        f"**애니메이션 URL**\n{structured_response['애니메이션_URL']}\n\n"
        f"**관련 법률 조항 및 해석**\n{structured_response['관련_법률']}\n\n"
        f"**해결 방안**\n{structured_response['해결_방안']}\n\n"
        f"**법적 리스크 및 고려 사항**\n{structured_response['법적_리스크']}\n\n"
        f"**결론**\n{structured_response['결론']}"
    )
    
    return formatted_response

def get_basic_llm():
    return ChatOpenAI(
        model="gpt-4o-mini-2024-07-18",
        temperature=0.3,
        openai_api_key=os.environ['OPENAI_API_KEY']
    )

# def load_memory(_):
#     return memory.load_memory_variables({})["chat_history"]
def load_memory(memory):
    chat_history = memory.load_memory_variables({})["chat_history"]
    return [msg.content for msg in chat_history]



def analyze_text(question):
    # from .conversation import ConversationManager, chat_history_collection
    try:
        # LLM 및 메모리 초기화
        memory = ConversationSummaryBufferMemory(
            llm=get_basic_llm(),
            max_token_limit=200,
            return_messages=True,
            memory_key="chat_history"  # 메모리 변수를 일치시킵니다.
        )
        conversation_manager = ConversationManager()

        if not question or question.strip() == "":
            raise ValueError("질문이 제공되지 않았습니다.")   
        # elif question or question.stirp().lower() == '종료':
        #     print('종료')
        #     # 대화 기록 저장
        #     chat_data = {
        #         "user": question,
        #         "AI_Model": "종료 명령어",
        #         "summary": "대화가 종료되었습니다."
        #     }
            
        #     conversation_manager.add_conversation(chat_data)

        #     # 메모리 초기화
        #     conversation_manager.clear_memory()

        #     break

        chain = (
            {
                "context": get_ret(),
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(lambda _: load_memory(memory))
            }
            | system_template
            | get_basic_llm()
        )

        result = chain.invoke(question)

        # 메모리에 대화 기록 저장
        memory.save_context({"user": question}, {"AI_Model": str(result)})

        # 메모리에서 실시간으로 요약한 대화 기록 추출
        chat_history = memory.load_memory_variables({})["chat_history"]

        # 요약된 대화 기록을 문자열로 변환
        chat_history_str = "".join([msg.content for msg in chat_history])   

        # 질문과 응답을 리스트에 추가
        chat_data = {
            "user": question,
            "AI_Model": str(result),
            "summary": chat_history_str
        }
        
        conversation_manager.add_conversation(chat_data)

        return result.content
                
    except Exception as e:
        print(f"텍스트 분석 중 오류가 발생했습니다: {str(e)}")
        return f"오류 발생: {str(e)}"


def analyze_video(file_path, seconds_per_frame=2):
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
            clip.audio.write_audiofile(audio_path, bitrate="32k")
            clip.audio.close()
        else:
            audio_path = None
    except Exception as e:
        audio_path = None
    clip.close()
    
    return base64Frames, audio_path

def summarize_video(base64Frames, audio_path):
    summary_text = ""
    if audio_path is not None:
        transcription = client.Audio.transcribe(
            model="whisper-1",
            file=open(audio_path, 'rb')
        )
        summary_text += transcription['text'] + "\n"

        response_both = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": video_msg},
                {"role": "user", "content": f"이건 비디오 영상의 오디오 텍스트: {transcription['text']}"},
                {"role": "user", "content": "이 영상은 오디오와 함께 여러 프레임을 포함하고 있습니다. 여기에는 프레임 이미지가 포함되지만 텍스트 설명만 제공합니다."}
            ],
            temperature=0.3,
            top_p=0.9
        )
        summary_text += response_both.choices[0].message.content + "\n"
    else:
        response_vis = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": video_msg},
                {"role": "user", "content": "이건 오디오가 없는 비디오 영상입니다. 아래 프레임을 기반으로 분석했습니다.  프레임 이미지가 포함되지만 텍스트 설명만 제공합니다."}
            ],
            temperature=0.3,
            top_p=0.9
        )
        summary_text += response_vis.choices[0].message.content + "\n"
    
    return summary_text

def analyze_video_with_interaction(file_path):
    base64Frames, audio_path = analyze_video(file_path, seconds_per_frame=2)
    video_summary = summarize_video(base64Frames, audio_path)
    # 영상 분석 결과 저장
    analysis_data = {
        "file_path": file_path,
        "summary": video_summary
    }
    video_analysis_collection.insert_one(analysis_data)
    return video_summary