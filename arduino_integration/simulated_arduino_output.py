import time
import random
import json
import os
import serial
import datetime


'''def generate_simulated_data():
    temperature = round(random.uniform(24, 30), 1)
    humidity = random.randint(55, 70)
    voltage = round(random.uniform(2.3, 2.6), 6)
    current = round(random.uniform(0.05, 0.09), 5)
    return temperature, humidity, voltage, current'''
    
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=2)

# Path to write the latest data for dashboard
DATA_PATH = os.path.join(os.path.dirname(__file__), 'latest_arduino_data.json')

try:
    while True:
        line = arduino.readline().decode('utf-8').strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                data = json.loads(line)
                # Add timestamp
                data["timestamp"] = datetime.datetime.now().isoformat()
                print(f"Vout: {data['voltage']:.2f} | Raw current: {data['current']:.2f}")
                print(json.dumps(data, indent=2))
                print()

                # Save to JSON file
                with open(DATA_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                
                time.sleep(2)
            except json.JSONDecodeError:
                print("Invalid JSON:", line)
except KeyboardInterrupt:
    print("Stopped by user.")