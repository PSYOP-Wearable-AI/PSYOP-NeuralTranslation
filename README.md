# PSYOP Neural Translation

A real-time OCR and translation application using Meta's NLLB and OpenAI's Whisper for field translation capabilities.

## Features

- **Real-time OCR**: Detects text from camera feed using EasyOCR
- **Speech-to-Text**: Transcribes audio using OpenAI Whisper
- **Neural Translation**: Translates text using Meta's NLLB model
- **Multi-language Support**: Supports 200+ languages
- **Optimized Performance**: Threaded processing for real-time performance
- **Modern UI**: Clean, dark-themed interface with live video feed

## Requirements

- Python 3.9+
- macOS (tested on macOS 25.0.0)
- Camera access
- Microphone access

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PSYOP-Wearable-AI/PSYOP-NeuralTranslation.git
cd PSYOP-NeuralTranslation
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install PortAudio (required for PyAudio):
```bash
brew install portaudio
```

## Usage

### Basic Usage
```bash
python realtime_translator.py
```

### Legacy Version
```bash
python live_translator.py
```

## Configuration

### Language Settings
- Select target language from the dropdown menu
- Supports 200+ languages including Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, and Hindi

### Performance Tuning
- OCR processes every 3rd frame for optimal performance
- Audio processing interval: 3 seconds
- Confidence threshold: 0.3 (adjustable)

## Architecture

### Components
- **AudioCaptureThread**: Real-time audio capture
- **WhisperTranscriptionThread**: Speech-to-text processing
- **NLLBTranslationThread**: Neural machine translation
- **OCRThread**: Optical character recognition
- **RealtimeTranslator**: Main UI and coordination

### Models Used
- **Whisper**: "tiny" model for real-time speech recognition
- **NLLB**: "facebook/nllb-200-distilled-600M" for translation
- **EasyOCR**: Multi-language text detection

## Performance Optimizations

- **Model Quantization**: Half-precision floating point (float16)
- **Threading**: Non-blocking UI with separate processing threads
- **Frame Skipping**: Intelligent frame processing
- **Memory Management**: Low CPU memory usage
- **GPU Acceleration**: Automatic GPU detection and usage

## File Structure

```
PSYOP-NeuralTranslation/
├── realtime_translator.py    # Main application (NLLB + Whisper)
├── live_translator.py        # Legacy version (MarianMT)
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── .gitignore              # Git ignore rules
```

## Dependencies

- `torch` - PyTorch framework
- `torchaudio` - Audio processing
- `transformers` - Hugging Face transformers
- `whisper-openai` - OpenAI Whisper
- `easyocr` - OCR engine
- `opencv-python` - Computer vision
- `PySide6` - GUI framework
- `numpy` - Numerical computing
- `pyaudio` - Audio I/O

## Troubleshooting

### Common Issues

1. **PortAudio Error**: Install PortAudio using `brew install portaudio`
2. **Model Loading**: Models download automatically on first run
3. **Camera Access**: Ensure camera permissions are granted
4. **Performance**: Reduce frame processing frequency if lag occurs

### Debug Mode
The application includes extensive debug output. Check console for:
- OCR detection results
- Translation progress
- Error messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Meta AI for NLLB translation model
- OpenAI for Whisper speech recognition
- EasyOCR for text detection
- Hugging Face for transformers library