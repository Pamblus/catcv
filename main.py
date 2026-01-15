#!/usr/bin/env python3
"""
MeowCV HTTPS Server с поддержкой папки assets
"""

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import base64
import os
import socket
import ssl
import subprocess
import sys
from datetime import datetime
import atexit

# ================= КОНФИГУРАЦИЯ =================
CONFIG = {
    'http_port': 3000,
    'https_port': 3010,
    'host': '0.0.0.0',
    'ssl_cert': 'cert.pem',
    'ssl_key': 'key.pem',
    'assets_folder': 'assets',
    'auto_create_cert': True,
    'ngrok_enabled': False,
    'ngrok_auth_token': '',
    'ngrok_domain': ''
}

# ================= ИНИЦИАЛИЗАЦИЯ =================

# Инициализация Flask
app = Flask(__name__)

# Инициализация Mediapipe
print("🎭 Инициализация Mediapipe...")
try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_faces=1,
        static_image_mode=False
    )
    print("✅ Mediapipe готов")
except Exception as e:
    print(f"❌ Ошибка Mediapipe: {e}")
    sys.exit(1)

# Создаем папки
os.makedirs("templates", exist_ok=True)
os.makedirs(CONFIG['assets_folder'], exist_ok=True)  # Используем assets

# ================= SSL СЕРТИФИКАТЫ =================

def create_self_signed_cert():
    """Создание самоподписанного SSL сертификата"""
    cert_file = CONFIG['ssl_cert']
    key_file = CONFIG['ssl_key']
    
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print("✅ SSL сертификаты уже существуют")
        return True
    
    print("🔐 Создание самоподписанного SSL сертификата...")
    try:
        # Создаем приватный ключ
        private_key = subprocess.run([
            'openssl', 'genrsa', '-out', key_file, '2048'
        ], capture_output=True, text=True)
        
        # Создаем CSR
        csr = subprocess.run([
            'openssl', 'req', '-new', '-key', key_file, '-out', 'csr.pem',
            '-subj', '/C=RU/ST=Moscow/L=Moscow/O=MeowCV/CN=meowcv.local'
        ], capture_output=True, text=True)
        
        # Создаем самоподписанный сертификат
        cert = subprocess.run([
            'openssl', 'x509', '-req', '-days', '365', '-in', 'csr.pem',
            '-signkey', key_file, '-out', cert_file
        ], capture_output=True, text=True)
        
        # Удаляем временный файл
        if os.path.exists('csr.pem'):
            os.remove('csr.pem')
        
        print("✅ SSL сертификаты созданы")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка создания SSL: {e}")
        print("⚠️  Запускаю без HTTPS (только HTTP)")
        return False

# ================= НАСТРОЙКИ ДЕТЕКЦИИ =================

THRESHOLDS = {
    'eye_opening': 0.025,
    'mouth_open': 0.03,
    'squinting': 0.018
}

def detect_expression(landmarks):
    """Определение выражения лица"""
    points = landmarks.landmark
    
    # Глаза
    left_eye = abs(points[159].y - points[145].y)
    right_eye = abs(points[386].y - points[374].y)
    avg_eye = (left_eye + right_eye) / 2
    
    # Рот
    mouth = abs(points[13].y - points[14].y)
    
    if avg_eye > THRESHOLDS['eye_opening']:
        return 'shock'
    elif mouth > THRESHOLDS['mouth_open']:
        return 'tongue'
    elif avg_eye < THRESHOLDS['squinting']:
        return 'glare'
    else:
        return 'default'

# ================= РОУТЫ FLASK =================

@app.route('/')
def index():
    return render_template('index_https.html')

@app.route(f'/{CONFIG["assets_folder"]}/<path:filename>')
def serve_assets(filename):
    """Сервим файлы из папки assets"""
    return send_from_directory(CONFIG['assets_folder'], filename)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(CONFIG['assets_folder'], 'favicon.ico', 
                               mimetype='image/vnd.microsoft.icon')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'success': False, 'error': 'Нет данных изображения'})
        
        # Декодируем base64
        if ',' in data['image']:
            image_data = data['image'].split(',')[1]
        else:
            image_data = data['image']
            
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'error': 'Не удалось декодировать изображение'})
        
        # Конвертируем в RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Детектируем лица
        results = face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            expression = detect_expression(landmarks)
            
            # Пути к картинкам в assets
            cat_urls = {
                'shock': f'/{CONFIG["assets_folder"]}/cat-shock.jpeg',
                'tongue': f'/{CONFIG["assets_folder"]}/cat-tongue.jpeg', 
                'glare': f'/{CONFIG["assets_folder"]}/cat-glare.jpeg',
                'default': f'/{CONFIG["assets_folder"]}/default-cat.jpeg'
            }
            
            return jsonify({
                'success': True,
                'expression': expression,
                'cat_image': cat_urls.get(expression, f'/{CONFIG["assets_folder"]}/default-cat.jpg'),
                'message': f'Обнаружено: {expression}'
            })
        else:
            return jsonify({
                'success': False,
                'expression': 'none',
                'message': 'Лицо не обнаружено'
            })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/update_thresholds', methods=['POST'])
def update_thresholds():
    try:
        data = request.json
        for key in ['eye_opening', 'mouth_open', 'squinting']:
            if key in data:
                THRESHOLDS[key] = float(data[key])
        
        return jsonify({
            'success': True, 
            'thresholds': THRESHOLDS,
            'message': 'Настройки обновлены'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/camera_status', methods=['GET'])
def camera_status():
    return jsonify({
        'success': True,
        'message': 'Сервер готов к работе',
        'requires_https': False,  # У нас есть HTTPS
        'assets_folder': CONFIG['assets_folder'],
        'https_available': True
    })

@app.route('/server_info', methods=['GET'])
def server_info():
    """Информация о сервере"""
    return jsonify({
        'success': True,
        'server': 'MeowCV HTTPS',
        'version': '2.0',
        'https_port': CONFIG['https_port'],
        'http_port': CONFIG['http_port'],
        'assets': CONFIG['assets_folder'],
        'protocols': ['http', 'https'],
        'time': datetime.now().isoformat()
    })

# ================= СОЗДАНИЕ ТЕСТОВЫХ ИЗОБРАЖЕНИЙ =================

def create_test_images():
    """Создание тестовых изображений котиков в assets"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        size = 300
        colors = {
            'shock': (255, 107, 107),
            'tongue': (78, 205, 196),
            'glare': (69, 183, 209),
            'default': (150, 206, 180)
        }
        
        texts = {
            'shock': '😲 ШОК!',
            'tongue': '😛 ЯЗЫК',
            'glare': '😒 ПРИЩУР',
            'default': '😊 НОРМА'
        }
        
        for name, color in colors.items():
            img = Image.new('RGB', (size, size), color)
            draw = ImageDraw.Draw(img)
            
            try:
                # Пробуем разные шрифты
                fonts_to_try = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    "arial.ttf"
                ]
                font = None
                for font_path in fonts_to_try:
                    try:
                        font = ImageFont.truetype(font_path, 40)
                        break
                    except:
                        continue
                
                if font is None:
                    font = ImageFont.load_default()
                    
            except:
                font = ImageFont.load_default()
            
            text = texts[name]
            # Простой расчет позиции
            text_width = len(text) * 20  # Примерная ширина
            text_height = 40
            position = ((size - text_width) // 2, (size - text_height) // 2)
            draw.text(position, text, fill="white", font=font)
            
            # Сохраняем в assets
            img.save(f'{CONFIG["assets_folder"]}/{name}-cat.jpg')
            print(f"✅ Создано: {CONFIG['assets_folder']}/{name}-cat.jpg")
            
    except Exception as e:
        print(f"⚠️  Не удалось создать изображения: {e}")
        # Создаем простые файлы-заглушки
        for name in ['shock', 'tongue', 'glare', 'default']:
            with open(f'{CONFIG["assets_folder"]}/{name}-cat.jpg', 'wb') as f:
                f.write(b'fake_image')
        print("⚠️  Созданы заглушки для изображений")

# ================= NGROK ИНТЕГРАЦИЯ =================

def start_ngrok_tunnel(port):
    """Запуск ngrok туннеля"""
    if not CONFIG['ngrok_enabled']:
        return None
    
    try:
        import requests
        from threading import Thread
        import time
        
        def ngrok_thread():
            try:
                # Устанавливаем ngrok если нет
                ngrok_path = '/usr/local/bin/ngrok'
                if not os.path.exists(ngrok_path):
                    print("📥 Установка ngrok...")
                    os.system('curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null')
                    os.system('echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list')
                    os.system('sudo apt update && sudo apt install ngrok -y')
                
                # Авторизация
                if CONFIG['ngrok_auth_token']:
                    os.system(f'ngrok config add-authtoken {CONFIG["ngrok_auth_token"]}')
                
                # Запуск туннеля
                cmd = f'ngrok http {port}'
                if CONFIG['ngrok_domain']:
                    cmd += f' --domain={CONFIG["ngrok_domain"]}'
                
                print(f"🚀 Запуск ngrok: {cmd}")
                os.system(cmd)
                
            except Exception as e:
                print(f"❌ Ошибка ngrok: {e}")
        
        # Запускаем в отдельном потоке
        thread = Thread(target=ngrok_thread, daemon=True)
        thread.start()
        
        # Даем время на запуск
        time.sleep(3)
        
        # Получаем URL туннеля
        try:
            response = requests.get('http://localhost:4040/api/tunnels')
            data = response.json()
            if data['tunnels']:
                public_url = data['tunnels'][0]['public_url']
                print(f"🌍 Ngrok URL: {public_url}")
                return public_url
        except:
            pass
            
    except Exception as e:
        print(f"⚠️  Ngrok не запущен: {e}")
    
    return None

# ================= АВТОМАТИЧЕСКИЙ ПОРТ =================

def get_available_port(start_port, max_tries=100):
    """Находит свободный порт"""
    for port in range(start_port, start_port + max_tries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result != 0:  # Порт свободен
                return port
        except:
            continue
    return start_port

# ================= ЗАПУСК СЕРВЕРА =================

def run_server():
    """Запуск сервера с поддержкой HTTPS"""
    
    print("\n" + "="*60)
    print("🐱 MEOWCV HTTPS SERVER v2.0")
    print("="*60)
    
    # Создаем тестовые изображения
    print("\n🖼️  Создание тестовых изображений...")
    create_test_images()
    
    # Находим свободные порты
    http_port = get_available_port(CONFIG['http_port'])
    https_port = get_available_port(CONFIG['https_port'])
    
    # Обновляем конфиг
    CONFIG['http_port'] = http_port
    CONFIG['https_port'] = https_port
    
    print(f"\n📦 Зависимости:")
    print(f"  Python: {np.__version__}")
    print(f"  OpenCV: {cv2.__version__}")
    print(f"  Mediapipe: {mp.__version__}")
    
    print(f"\n🌐 ДОСТУПНЫЕ АДРЕСА:")
    print(f"  HTTPS: https://you.ip.address:{https_port}")
    print(f"  HTTP:  http://you.ip.address:{http_port}")
    print(f"  Локально: https://localhost:{https_port}")
    print(f"  Папка с картинками: /{CONFIG['assets_folder']}/")
    
    print(f"\n🖼️  Картинки в assets:")
    if os.path.exists(CONFIG['assets_folder']):
        for img in os.listdir(CONFIG['assets_folder']):
            if img.endswith(('.jpg', '.jpeg', '.png')):
                print(f"  • {CONFIG['assets_folder']}/{img}")
    
    # Создаем SSL сертификаты
    ssl_available = False
    if CONFIG['auto_create_cert']:
        ssl_available = create_self_signed_cert()
    
    # Запускаем ngrok если нужно
    ngrok_url = None
    if CONFIG['ngrok_enabled']:
        ngrok_url = start_ngrok_tunnel(https_port)
        if ngrok_url:
            print(f"\n🌍 NGROK HTTPS: {ngrok_url}")
    
    print("\n" + "="*60)
    print("🚀 Запуск серверов...")
    print("="*60)
    
    # Функция для запуска HTTP сервера в отдельном потоке
    def run_http_server():
        try:
            from flask import Flask
            http_app = Flask(__name__)
            
            @http_app.route('/')
            def http_redirect():
                return f'''
                <!DOCTYPE html>
                <html>
                <head>
                    <meta http-equiv="refresh" content="0; url=https://you.ip.address:{https_port}">
                    <title>Redirecting to HTTPS...</title>
                </head>
                <body>
                    <h1>Redirecting to HTTPS...</h1>
                    <p>If not redirected, <a href="https://you.ip.address:{https_port}">click here</a>.</p>
                </body>
                </html>
                '''
            
            print(f"📡 HTTP сервер запущен на порту {http_port}")
            http_app.run(host=CONFIG['host'], port=http_port, debug=False, threaded=True, use_reloader=False)
        except Exception as e:
            print(f"⚠️  HTTP сервер не запущен: {e}")
    
    # Запускаем HTTP сервер в фоне
    import threading
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    
    # Основной HTTPS сервер
    try:
        if ssl_available and os.path.exists(CONFIG['ssl_cert']) and os.path.exists(CONFIG['ssl_key']):
            # Контекст SSL
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(CONFIG['ssl_cert'], CONFIG['ssl_key'])
            
            print(f"\n✅ HTTPS сервер запущен!")
            print(f"🔐 Используется SSL: {CONFIG['ssl_cert']}")
            print(f"📱 Откройте в браузере: https://you.ip.address:{https_port}")
            print(f"\n🎭 Готов к детекции эмоций!")
            print("="*60)
            
            app.run(
                host=CONFIG['host'],
                port=https_port,
                debug=True,
                threaded=True,
                use_reloader=False,
                ssl_context=context
            )
        else:
            print("⚠️  SSL сертификаты не найдены, запускаю HTTP")
            print(f"📱 Откройте: http://you.ip.address:{http_port}")
            app.run(
                host=CONFIG['host'],
                port=http_port,
                debug=True,
                threaded=True,
                use_reloader=False
            )
            
    except KeyboardInterrupt:
        print("\n\n👋 Остановка сервера...")
    except Exception as e:
        print(f"\n❌ Ошибка сервера: {e}")
        import traceback
        traceback.print_exc()

# ================= ТОЧКА ВХОДА =================

if __name__ == '__main__':
    run_server()
