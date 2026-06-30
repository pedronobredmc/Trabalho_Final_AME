[README.md](https://github.com/user-attachments/files/29490080/README.md)
# Detecção de Pessoa com TinyML no ESP32-CAM

Projeto final da disciplina **Aprendizado de Máquina** — UFC Campus Quixadá

Classificador binário (pessoa / sem pessoa) treinado com Transfer Learning (MobileNetV2 α=0.35), quantizado para INT8 e embarcado no ESP32-CAM usando TensorFlow Lite for Microcontrollers.

---

## Estrutura do Repositório

```
/
├── README.md
├── requirements.txt
├── notebook/
│   └── teste96.ipynb           # Treino, avaliação e conversão TFLite
├── model/
│   ├── modelo_industrial_int8.tflite   # Modelo quantizado INT8 (771 KB)
│   └── modelo_industrial.h             # Array C para embarcar no ESP32
├── pico/
│   ├── esp32cam_person_detection.ino   # Sketch Arduino (ESP32-CAM)
│   └── INSTRUÇÕES.md                   # Guia de embarque passo a passo
└── data/
    └── README.md                       # Instruções para obter o dataset
```

---

## Descrição do Projeto

O objetivo é rodar inferência de visão computacional **diretamente no hardware**, sem depender de servidor ou nuvem (Edge AI / TinyML).

**Pipeline completo:**
1. Treino no Google Colab com MobileNetV2 α=0.35 (entrada 96×96)
2. Quantização INT8: 9.3 MB → 771 KB (redução de 69.8%)
3. Conversão para array C (`.h`)
4. Upload no ESP32-CAM via Arduino IDE

**Dataset:** `dataset_industrial_limto` — contexto industrial, 2 classes (person / no-person), split 80/20, batch size 32.

---

## Como Instalar

### Pré-requisitos

- Python 3.9+
- Google Colab (recomendado) ou ambiente local com GPU
- Arduino IDE 2.x
- ESP32-CAM AI-Thinker

### Ambiente Python

```bash
pip install -r requirements.txt
```

---

## Como Rodar o Notebook

1. Abre o arquivo `notebook/teste96.ipynb` no Google Colab
2. Faz upload do dataset `dataset_industrial_limto` para `/content/`
3. Executa todas as células em ordem (Runtime → Executar tudo)
4. Os arquivos gerados serão baixados automaticamente:
   - `modelo_industrial_int8.tflite`
   - `modelo_industrial.h`

---

## Como Gravar no ESP32-CAM

1. Instala o suporte ao ESP32 na Arduino IDE:
   - Arquivo → Preferências → URLs adicionais:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
2. Instala a biblioteca **TensorFlow Lite Micro for Espressif chips**
3. Cria pasta `esp32cam_person_detection/` com os arquivos:
   - `pico/esp32cam_person_detection.ino`
   - `model/modelo_industrial.h`
4. Seleciona a placa: **AI Thinker ESP32-CAM**
5. Partition Scheme: **Huge APP (3MB No OTA)** ← obrigatório
6. Conecta GPIO0 ao GND → clica Upload → aperta RESET
7. Abre o Monitor Serial em 115200 baud

Veja o guia completo em `pico/INSTRUÇÕES.md`.

---

## Resultados

| Versão | Tamanho | ESP32-CAM |
|--------|---------|-----------|
| MobileNetV2 1.0 Float32 | 9.3 MB | ❌ Não cabe |
| MobileNetV2 0.35 Float32 | 2.2 MB | ⚠️ Apertado |
| MobileNetV2 0.35 INT8 | **771 KB** | ✅ Cabe folgado |

Saída no Monitor Serial:
```
✅ PESSOA DETECTADA   (confiança: 87.3%)
❌ SEM PESSOA         (confiança: 92.1%)
```

---

## Links

- 📄 **Artigo:** https://www.overleaf.com/4418994722nwhhfbtjqzcd#6f5bf2
- 📊 **Slides:** `apresentacao_tinyml.pptx` (na raiz do repositório)
- 🔗 **Repositório:** https://github.com/pedronobredmc/Trabalho_Final_AME

---

## Autores

UFC Campus Quixadá — Disciplina: Aprendizado de Máquina
