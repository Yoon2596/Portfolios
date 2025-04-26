# app.py

from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import os
import uuid
from conversation import ConversationManager, chat_history_collection  # 이 부분을 추가
from user_service import save_user, check_user
from analysis_service import analyze_text, analyze_video_with_interaction
from retriever import get_ret

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize memory when the app starts
conversation_manager = ConversationManager()
conversation_manager.clear_memory()

@app.route('/save_conversations', methods=['POST'])
def save_conversations():
    global conversation_manager
    if conversation_manager is None:
        # from py.conversation import ConversationManager, chat_history_collection
        conversation_manager = ConversationManager()
    
    message, status = conversation_manager.save_conversations(chat_history_collection)
    return jsonify({'message': message}), status

# 업로드 폴더 설정
app.config['UPLOAD_FOLDER'] = 'uploads/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login_main')
def login_main():
    return render_template('login_main.html')

@app.route('/login', methods=['POST'])
def login():
    id = request.form['id']
    password = request.form['password']
    nickname = check_user(id, password)
    if nickname:
        return redirect(url_for('index_login', nickname=nickname))
    else:
        return "로그인 실패. 다시 시도해 주세요."

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register_user():
    id = request.form['id']
    password = request.form['password']
    nickname = request.form['nickname']
    email = request.form['email']
    birthdate = request.form['birthdate']
    save_user(id, password, nickname, email, birthdate)
    return redirect(url_for('login_main'))

@app.route('/index_login')
def index_login():
    nickname = request.args.get('nickname')
    return render_template('index_login.html', nickname=nickname)

@app.route('/ask', methods=['POST'])
def ask():
    if 'video' in request.files:
        try:
            video_file = request.files['video']
            unique_filename = str(uuid.uuid4()) + os.path.splitext(video_file.filename)[1]
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            video_file.save(video_path)
            response_text = "영상이 업로드 되었습니다. 분석을 시작합니다. 잠시만 기다려주세요..."
            video_analysis = analyze_video_with_interaction(video_path)
            response_text += "\n\n분석 결과: " + video_analysis
            os.remove(video_path)
        except Exception as e:
            return jsonify({'answer': f"오류가 발생했습니다: {str(e)}"}), 500
    else:
        try:
            user_input = request.form['question']
            response_text = analyze_text(user_input)
        except Exception as e:
            return jsonify({'answer': f"오류가 발생했습니다: {str(e)}"}), 500

    return jsonify({'answer': response_text})

@app.route('/logout')
def logout():
    session.pop('nickname', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(port=9199)
