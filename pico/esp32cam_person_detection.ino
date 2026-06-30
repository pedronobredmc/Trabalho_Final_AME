// ============================================================
// Detecção de Pessoa — ESP32-CAM + TensorFlow Lite
// Modelo: MobileNetV2 0.35 | Entrada: 96x96 INT8
// ============================================================
#include <Arduino.h>
#include "esp_camera.h"
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "modelo_industrial.h"

// ============================================================
// CONFIGURAÇÃO DE PINOS — AI-Thinker ESP32-CAM
// ============================================================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define LED_FLASH         4

// ============================================================
// CONFIGURAÇÃO DO TFLITE
// ============================================================
const int IMG_WIDTH  = 96;
const int IMG_HEIGHT = 96;

constexpr int kTensorArenaSize = 1024 * 1024;
uint8_t *tensor_arena;  // ponteiro — será alocado na PSRAM

tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor  = nullptr;
TfLiteTensor* output_tensor = nullptr;

// ============================================================
// INICIALIZA CÂMERA
// ============================================================
bool inicializar_camera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size   = FRAMESIZE_96X96;
  config.fb_count     = 1;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Erro ao inicializar câmera: 0x%x\n", err);
    return false;
  }
  return true;
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  pinMode(LED_FLASH, OUTPUT);
  digitalWrite(LED_FLASH, LOW);

  Serial.println("\n=== Sistema de Detecção de Pessoa ===");

  // Aloca tensor arena na PSRAM
  tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
  if (!tensor_arena) {
    Serial.println("ERRO: Falha ao alocar PSRAM! Verifique se PSRAM está habilitada em Tools.");
    while (true) delay(1000);
  }
  Serial.println("✓ PSRAM OK");

  // Inicializa câmera
  if (!inicializar_camera()) {
    Serial.println("ERRO: Câmera não inicializada. Reiniciando...");
    delay(3000);
    ESP.restart();
  }
  Serial.println("✓ Câmera OK");

  // Carrega o modelo
  model = tflite::GetModel(modelo_industrial);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERRO: Versão do modelo incompatível!");
    return;
  }

  // Cria o interpretador
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERRO: Falha ao alocar tensores!");
    while(true);
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.println("✓ Modelo TFLite carregado");
  Serial.printf("✓ Arena usada: %d bytes\n", interpreter->arena_used_bytes());
  Serial.println("\nIniciando detecção...\n");
}

// ============================================================
// LOOP PRINCIPAL
// ============================================================
void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Erro ao capturar frame.");
    delay(500);
    return;
  }

  uint16_t* src = (uint16_t*)fb->buf;
  int8_t* dst   = input_tensor->data.int8;

  for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
    uint16_t pixel = src[i];
    int r = ((pixel >> 11) & 0x1F) << 3;
    int g = ((pixel >> 5)  & 0x3F) << 2;
    int b = (pixel & 0x1F) << 3;
    dst[i * 3 + 0] = (int8_t)(r - 128);
    dst[i * 3 + 1] = (int8_t)(g - 128);
    dst[i * 3 + 2] = (int8_t)(b - 128);
  }

  esp_camera_fb_return(fb);

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Erro na inferência.");
    return;
  }

  float score = (output_tensor->data.int8[0] - output_tensor->params.zero_point)
                * output_tensor->params.scale;

  if (score > 0.5) {
    Serial.printf("✅ PESSOA DETECTADA   (confiança: %.1f%%)\n", score * 100);
    digitalWrite(LED_FLASH, HIGH);
  } else {
    Serial.printf("❌ SEM PESSOA         (confiança: %.1f%%)\n", (1 - score) * 100);
    digitalWrite(LED_FLASH, LOW);
  }

  delay(500);
}
