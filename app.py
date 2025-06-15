import cv2
from ultralytics import YOLO
from telegram import Bot
import requests
import time
from threading import Thread
from flask import Flask, render_template
from flask_socketio import SocketIO
import os

TELEGRAM_TOKEN = "7894501731:AAGeusk3FBoSWzN093z3TFq0iYJT3A4R4cs"
TELEGRAM_CHAT_ID = "@Taratep"
ESP32_API_URL = "http://192.168.1.100/mq2"

bot = Bot(token=TELEGRAM_TOKEN)
model = YOLO("best.pt")

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

alert_sent = False

def send_telegram_message(text):
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
        print("ส่งข้อความแจ้งเตือนไปที่ Telegram แล้ว")
    except Exception as e:
        print("ส่งข้อความ Telegram ไม่สำเร็จ:", e)

def get_mq2_value():
    try:
        resp = requests.get(ESP32_API_URL, timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            return float(data.get("mq2", 0))
    except Exception as e:
        print("ไม่สามารถอ่านค่า MQ2 จาก ESP32:", e)
    return 0

def check_fire_and_smoke(frame):
    results = model(frame)[0]

    fire_found = False
    smoke_found = False

    if results.boxes:
        class_ids = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_names = results.names

        for cls, conf in zip(class_ids, confidences):
            name = class_names[int(cls)]
            if conf < 0.5:
                continue
            if name == "fire":
                fire_found = True
            elif name == "smoke":
                smoke_found = True

    return fire_found, smoke_found

def camera_thread():
    global alert_sent
    # บน server render.com ไม่มีเว็บแคม ให้ลองใช้ไฟล์วิดีโอแทน หรือ IP Camera URL
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("video.mp4")  # แก้ตามไฟล์วิดีโอ หรือ IP camera URL

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # รีวิดีโอวนซ้ำ
            continue

        mq2_value = get_mq2_value()
        fire_found, smoke_found = check_fire_and_smoke(frame)

        status = {
            "mq2": mq2_value,
            "fire": fire_found,
            "smoke": smoke_found
        }

        socketio.emit("status_update", status)

        if (mq2_value > 300 or (fire_found and smoke_found)) and not alert_sent:
            send_telegram_message(f"⚠️ แจ้งเตือน! พบความเสี่ยงไฟไหม้! MQ2={mq2_value:.2f}, Fire={fire_found}, Smoke={smoke_found}")
            alert_sent = True

        time.sleep(0.5)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    Thread(target=camera_thread, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)
