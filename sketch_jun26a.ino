#include <ArduinoJson.h>
#include <DHT.h>

// ---- Sensor Configuration ----
#define DHTPIN 2
#define DHTTYPE DHT11
#define ACS_PIN A0

DHT dht(DHTPIN, DHTTYPE);

// ---- Calibration Constants ----
const float ZERO_CURRENT_VOLTAGE = 2.495;     // Measured no-load voltage from getAverageVoltage()
const float SENSITIVITY = 0.185;              // ACS712 5A version = 185mV/A
const int NUM_SAMPLES = 50;                  // For analog noise smoothing
const float MIN_CURRENT_THRESHOLD = 0.07;     // Ignore noise below 70mA

// ---- Data transmission interval ----
const unsigned long TRANSMISSION_INTERVAL = 5000;  // 5 seconds
unsigned long lastTransmission = 0;

// ---- JSON document for data ----
StaticJsonDocument<512> jsonDoc;

void setup() {
  Serial.begin(9600);
  dht.begin();
  
  Serial.println("ENMOS Arduino Sensor System Ready!");
  Serial.println("Sending data every 5 seconds...");
  Serial.println();
}

void loop() {
  // Check if it's time to send data
  if (millis() - lastTransmission >= TRANSMISSION_INTERVAL) {
    // Read sensor data
    readSensorData();
    
    // Send data via Serial (JSON format)
    sendDataViaSerial();
    
    lastTransmission = millis();
  }
  
  // Small delay to prevent overwhelming the system
  delay(100);
}

// ---- Read Averaged Voltage from ACS712 ----
float getAverageVoltage() {
  float total = 0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    int reading = analogRead(ACS_PIN);
    float voltage = reading * 5.0 / 1023.0;
    total += voltage;
    delay(2); // fast sampling
  }
  return total / NUM_SAMPLES;
}

void readSensorData() {
  // Clear previous data
  jsonDoc.clear();
  
  // Add timestamp
  jsonDoc["timestamp"] = getTimestamp();
  
  // ---- Read temperature and humidity ----
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("DHT11 read error.");
    // Set default values if sensor fails
    temperature = 25.0;
    humidity = 50.0;
  }
  
  // Add temperature and humidity to JSON
  jsonDoc["temperature"] = temperature;
  jsonDoc["humidity"] = humidity;

  // ---- Read and calculate current ----
  float voltage = getAverageVoltage();
  float current = (voltage - ZERO_CURRENT_VOLTAGE) / SENSITIVITY;
  current = abs(current); // remove negative sign

  // Apply dead zone threshold to eliminate small drift
  if (current < MIN_CURRENT_THRESHOLD) {
    current = 0.0;
  }
  
  // Add current and voltage to JSON
  jsonDoc["current"] = current;
  jsonDoc["voltage"] = voltage;
  
  // ---- Calculate power ----
  float power = voltage * current;
  jsonDoc["power"] = power;
  
  // ---- Simulate other sensors for ENMOS compatibility ----
  // These can be replaced with real sensors later
  jsonDoc["vibration"] = random(0, 100) / 100.0;  // 0.0 to 1.0 g
  jsonDoc["pressure"] = 1000.0 + random(-50, 50);  // ~1000 hPa
  jsonDoc["viscosity"] = 45.0 + random(-5, 5);     // ~45 cP

  // ---- Debug Print ----
  Serial.println("=== Sensor Data ===");
  Serial.print("Temperature: "); Serial.print(temperature); Serial.println("°C");
  Serial.print("Humidity: "); Serial.print(humidity); Serial.println("%");
  Serial.print("Vout: "); Serial.print(voltage); Serial.println("V");
  Serial.print("Current: "); Serial.print(current); Serial.println("A");
  Serial.print("Power: "); Serial.print(power); Serial.println("W");
  Serial.println();
}

String getTimestamp() {
  // For Arduino, we'll use millis() based timestamp
  unsigned long seconds = millis() / 1000;
  unsigned long hours = seconds / 3600;
  unsigned long minutes = (seconds % 3600) / 60;
  seconds = seconds % 60;
  
  return String(hours) + ":" + String(minutes) + ":" + String(seconds);
}

void sendDataViaSerial() {
  // Send JSON data via Serial for Python script to read
  Serial.println("=== ENMOS_DATA_START ===");
  serializeJson(jsonDoc, Serial);
  Serial.println();
  Serial.println("=== ENMOS_DATA_END ===");
}

/*
 * Alternative: Ethernet Version (if you have Ethernet shield)
 */

/*
#include <Ethernet.h>
#include <EthernetClient.h>
#include <SPI.h>

byte mac[] = {0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED};
IPAddress serverIP(192, 168, 1, 20);  // Your PC's IP
EthernetClient client;

void setupEthernet() {
  if (Ethernet.begin(mac) == 0) {
    Serial.println("Failed to configure Ethernet using DHCP");
    return;
  }
  Serial.print("Ethernet IP: ");
  Serial.println(Ethernet.localIP());
}

void sendDataViaEthernet() {
  if (client.connect(serverIP, 5001)) {
    String jsonString;
    serializeJson(jsonDoc, jsonString);
    
    client.println("POST /api/arduino/data HTTP/1.1");
    client.println("Host: 192.168.1.20:5001");
    client.println("Content-Type: application/json");
    client.print("Content-Length: ");
    client.println(jsonString.length());
    client.println();
    client.println(jsonString);
    
    while (client.connected()) {
      String line = client.readStringUntil('\n');
      if (line == "\r") {
        break;
      }
    }
    
    while (client.available()) {
      String line = client.readString();
      Serial.println(line);
    }
    
    client.stop();
  }
}
*/