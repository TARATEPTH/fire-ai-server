from flask import Flask, request
import os
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/')
def home():
    return "🔥 Fire AI Server Ready"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        nparr = np.frombuffer(request.data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_fire = np.array([18, 50, 50])
        upper_fire = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_fire, upper_fire)
        fire_pixels = cv2.countNonZero(mask)

        if fire_pixels > 500:
            return "🔥 FIRE DETECTED"
        else:
            return "✅ SAFE"
    except Exception as e:
        return f"Error: {e}", 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # ใช้พอร์ตจาก Render หรือ 5000
    app.run(host='0.0.0.0', port=port)
