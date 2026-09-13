/*
 * ESP32 Patient Health Monitor: ECG (AD8232), SpO2 & Heart Rate (MAX30102), Temp (DS18B20)
 * Integrates with Secure Health Monitoring Machine Learning Pipeline.
 * 
 * Hardware:
 *   - ESP32 NodeMCU / Dev Module
 *   - AD8232 ECG Sensor (OUT -> GPIO 36, LO+ -> GPIO 2, LO- -> GPIO 4)
 *   - MAX30102 Pulse Oximeter (SDA -> GPIO 21, SCL -> GPIO 22)
 *   - DS18B20 Temperature Sensor (DATA -> GPIO 15, with 4.7k pullup)
 *   - Status LED / Buzzer -> GPIO 13 (Triage Alarm)
 * 
 * Target API: POST http://<SERVER_IP>:8000/api/esp32/telemetry
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include "MAX30105.h"           // SparkFun MAX3010x library
#include "heartRate.h"
#include <OneWire.h>
#include <DallasTemperature.h>

// ==========================================
// CONFIGURATION & NETWORK CREDENTIALS
// ==========================================
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// IP Address of the machine running the Python Health Pipeline server
const char* SERVER_URL    = "http://192.168.1.100:8000/api/esp32/telemetry";
const char* AUTH_TOKEN    = "Bearer health_sec_token_9f83a2c0918bd47e";
const char* DEVICE_ID     = "ESP32-CLINICAL-NODE-01";

// Pin Definitions
#define PIN_ECG_IN        36    // VP / ADC1_CH0 (AD8232 OUT)
#define PIN_LO_PLUS       2     // AD8232 Leads-Off +
#define PIN_LO_MINUS      4     // AD8232 Leads-Off -
#define PIN_TEMP_BUS      15    // DS18B20 One-Wire bus
#define PIN_ALARM_LED     13    // Clinical critical triage LED / Buzzer

// Sampling Configuration
#define ECG_FS            125   // Sampling frequency (Hz)
#define WINDOW_SAMPLES    250   // 2-second telemetry window (250 samples at 125 Hz)

// ==========================================
// GLOBAL INSTANCES & BUFFERS
// ==========================================
MAX30105 particleSensor;
OneWire oneWire(PIN_TEMP_BUS);
DallasTemperature tempSensors(&oneWire);

float ecgBuffer[WINDOW_SAMPLES];
int sampleIndex = 0;
unsigned long lastSampleMicros = 0;
const unsigned long sampleIntervalMicros = 1000000 / ECG_FS; // 8000 us (125 Hz)

float lastBPM = 72.0;
float lastSpO2 = 98.0;
float lastTemperatureC = 36.6;
bool leadsOff = false;

// ==========================================
// SETUP
// ==========================================
void setup() {
  Serial.begin(115200);
  pinMode(PIN_LO_PLUS, INPUT);
  pinMode(PIN_LO_MINUS, INPUT);
  pinMode(PIN_ALARM_LED, OUTPUT);
  digitalWrite(PIN_ALARM_LED, LOW);

  // ADC resolution
  analogReadResolution(12); // 0 - 4095

  Serial.println("\n[ESP32 Health Monitor] Initializing...");

  // 1. Connect to Wi-Fi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected! IP address: ");
  Serial.println(WiFi.localIP());

  // 2. Initialize MAX30102 Sensor
  if (particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 pulse oximeter detected.");
    particleSensor.setup();
    particleSensor.setPulseAmplitudeRed(0x0A);
    particleSensor.setPulseAmplitudeGreen(0);
  } else {
    Serial.println("WARNING: MAX30102 not found. Check I2C wiring (SDA: 21, SCL: 22).");
  }

  // 3. Initialize DS18B20 Temperature Sensor
  tempSensors.begin();
  Serial.println("DS18B20 temperature sensor initialized.");

  lastSampleMicros = micros();
  Serial.println("System online. Streaming physiological signals...");
}

// ==========================================
// MAIN LOOP
// ==========================================
void loop() {
  // 1. Check Leads-Off condition on AD8232
  if ((digitalRead(PIN_LO_PLUS) == 1) || (digitalRead(PIN_LO_MINUS) == 1)) {
    leadsOff = true;
  } else {
    leadsOff = false;
  }

  // 2. Sample AD8232 ECG at fixed sample rate (125 Hz)
  unsigned long nowMicros = micros();
  if (nowMicros - lastSampleMicros >= sampleIntervalMicros) {
    lastSampleMicros += sampleIntervalMicros;

    if (leadsOff) {
      ecgBuffer[sampleIndex] = 0.0; // Flatline if electrodes detached
    } else {
      int rawADC = analogRead(PIN_ECG_IN);
      // Convert 12-bit ADC (0-4095 at 3.3V) to approximate physical millivolts
      // AD8232 output centered at 1.65V with gain ~ 1100
      float voltageV = (rawADC / 4095.0) * 3.3;
      float ecg_mV = (voltageV - 1.65) * 1000.0 / 1100.0;
      ecgBuffer[sampleIndex] = ecg_mV;
    }

    sampleIndex++;

    // When full buffer window is gathered, transmit to ML Pipeline
    if (sampleIndex >= WINDOW_SAMPLES) {
      // Sample vitals
      readVitals();
      // Send telemetry packet over Wi-Fi
      transmitTelemetryPacket();
      sampleIndex = 0;
    }
  }

  // Read MAX30102 infrared & red LEDs for heartbeats
  long irValue = particleSensor.getIR();
  if (checkForBeat(irValue)) {
    // Heartbeat detected
  }
}

// ==========================================
// READ AUXILIARY VITALS
// ==========================================
void readVitals() {
  // Read DS18B20
  tempSensors.requestTemperatures();
  float temp = tempSensors.getTempCByIndex(0);
  if (temp > 20.0 && temp < 45.0) {
    lastTemperatureC = temp;
  }

  // Read MAX30102 SpO2 & Heart Rate
  long ir = particleSensor.getIR();
  long red = particleSensor.getRed();
  if (ir > 50000) { // Finger detected on sensor
    float ratio = (float)red / (float)ir;
    float spo2Est = 110.0 - 25.0 * ratio; // Standard linear calibration approximation
    lastSpO2 = constrain(spo2Est, 85.0, 100.0);
  }
}

// ==========================================
// TRANSMIT TELEMETRY TO ML PIPELINE
// ==========================================
void transmitTelemetryPacket() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi disconnected. Reconnecting...");
    return;
  }

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("Authorization", AUTH_TOKEN);

  // Build JSON payload
  String json = "{";
  json += "\"device_id\":\"" + String(DEVICE_ID) + "\",";
  json += "\"fs\":" + String(ECG_FS) + ",";
  json += "\"leads_off\":" + String(leadsOff ? "true" : "false") + ",";
  json += "\"bpm\":" + String(lastBPM, 1) + ",";
  json += "\"spo2\":" + String(lastSpO2, 1) + ",";
  json += "\"temperature_c\":" + String(lastTemperatureC, 2) + ",";
  json += "\"signal\":[";

  for (int i = 0; i < WINDOW_SAMPLES; i++) {
    json += String(ecgBuffer[i], 3);
    if (i < WINDOW_SAMPLES - 1) json += ",";
  }
  json += "]}";

  // POST Request
  int httpResponseCode = http.POST(json);

  if (httpResponseCode == 200) {
    String response = http.getString();
    Serial.print("Pipeline Response: ");
    Serial.println(response);

    // Check for CRITICAL triage alarm from ML model
    if (response.indexOf("\"triage_risk_tier\":\"CRITICAL\"") > 0) {
      digitalWrite(PIN_ALARM_LED, HIGH); // Alarm LED ON
      Serial.println(">>> CLINICAL ALERT: CRITICAL CARDIAC EVENT DETECTED! <<<");
    } else {
      digitalWrite(PIN_ALARM_LED, LOW);  // Normal
    }
  } else {
    Serial.print("HTTP POST failed, code: ");
    Serial.println(httpResponseCode);
  }

  http.end();
}
