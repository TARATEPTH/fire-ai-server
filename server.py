import cv2
from ultralytics import YOLO
from telegram import Bot
import requests
import time
import gdown

file_url = "https://drive.google.com/uc?id=1RZOFqKKIYYDR2vcF43TS05CvPfAdj902"
output = "best.pt"
gdown.download(file_url, output, quiet=False)

# ตั้งค่า Telegram Bot
TELEGRAM_TOKEN = "7894501731:AAGeusk3FBoSWzN093z3TFq0iYJT3A4R4cs"
TELEGRAM_CHAT_ID = "@Taratep"  # เปลี่ยนเป็น chat ID หรือ username ของคุณ
ESP32_API_URL = "http://192.168.1.100/mq2"  # URL API ESP32 สำหรับดึงค่า MQ2

bot = Bot(token=TELEGRAM_TOKEN)
model = YOLO("best.pt")

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
            data = resp.json()  # สมมุติ ESP32 ส่ง JSON {"mq2": ค่า}
            return float(data.get("mq2", 0))
    except Exception as e:
        print("ไม่สามารถอ่านค่า MQ2 จาก ESP32:", e)
    return 0

def check_fire_and_smoke(frame):
    results = model(frame)[0]

    if not results.boxes:
        return False, False

    class_ids = results.boxes.cls.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    class_names = results.names

    fire_found = False
    smoke_found = False

    for cls, conf in zip(class_ids, confidences):
        name = class_names[int(cls)]
        if conf < 0.5:
            continue
        if name == "fire":
            fire_found = True
        elif name == "smoke":
            smoke_found = True

    return fire_found, smoke_found

def main():
    cap = cv2.VideoCapture(0)
    alert_sent = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mq2_value = get_mq2_value()
        fire_found, smoke_found = check_fire_and_smoke(frame)

        print(f"MQ2={mq2_value:.2f}, Fire={fire_found}, Smoke={smoke_found}")

        # เงื่อนไขแจ้งเตือน
        if (mq2_value > 300 or (fire_found and smoke_found)) and not alert_sent:
            send_telegram_message(f"⚠️ แจ้งเตือน! พบความเสี่ยงไฟไหม้! MQ2={mq2_value:.2f}, Fire={fire_found}, Smoke={smoke_found}")
            alert_sent = True

        # แสดงภาพ
        cv2.imshow("Fire & Smoke Detection", frame)

        # กด q เพื่อออก
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
