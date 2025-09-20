import sys
import cv2
import torch
import numpy as np
import threading
import time
import queue
import pyaudio
import wave
import tempfile
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                               QWidget, QLabel, QPushButton, QTextEdit, QComboBox, 
                               QSlider, QCheckBox, QProgressBar, QGroupBox)
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import QTimer, Qt, QThread, Signal, QMutex
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import easyocr

class AudioCaptureThread(QThread):
    """Thread for capturing audio in real-time"""
    audio_ready = Signal(np.ndarray)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
    def run(self):
        self.running = True
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            self.stream.start_stream()
            
            while self.running:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Audio capture error: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.audio.terminate()
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.running:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.audio_ready.emit(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def stop(self):
        self.running = False
        self.wait()

class WhisperTranscriptionThread(QThread):
    """Thread for Whisper speech-to-text processing"""
    transcription_ready = Signal(str, str)  # text, language
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.audio_buffer = []
        self.buffer_duration = 3.0  # seconds
        self.sample_rate = 16000
        self.model = None
        self.mutex = QMutex()
        
    def run(self):
        # Load Whisper model with optimizations
        try:
            # Use the smallest model for real-time performance
            self.model = whisper.load_model("tiny", device="cpu")  # Use CPU for better stability
            print("Whisper tiny model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            return
    
    def add_audio(self, audio_data):
        """Add audio data to buffer"""
        self.mutex.lock()
        self.audio_buffer.extend(audio_data)
        
        # Keep only last N seconds of audio
        max_samples = int(self.buffer_duration * self.sample_rate)
        if len(self.audio_buffer) > max_samples:
            self.audio_buffer = self.audio_buffer[-max_samples:]
        
        self.mutex.unlock()
    
    def process_audio(self):
        """Process audio buffer with Whisper"""
        if not self.model or not self.running:
            return
            
        self.mutex.lock()
        if len(self.audio_buffer) < self.sample_rate:  # Less than 1 second
            self.mutex.unlock()
            return
            
        audio_array = np.array(self.audio_buffer, dtype=np.float32) / 32768.0
        self.mutex.unlock()
        
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_array,
                language=None,  # Auto-detect language
                task="transcribe",
                fp16=False
            )
            
            text = result["text"].strip()
            detected_language = result.get("language", "unknown")
            
            if text and len(text) > 2:
                self.transcription_ready.emit(text, detected_language)
                
        except Exception as e:
            print(f"Whisper transcription error: {e}")
    
    def stop(self):
        self.running = False
        self.wait()

class NLLBTranslationThread(QThread):
    """Thread for NLLB translation processing"""
    translation_ready = Signal(str, str, str)  # original, translated, source_lang
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.translation_queue = queue.Queue()
        
    def run(self):
        # Load NLLB model with optimizations
        try:
            # Use smaller, faster model
            model_name = "facebook/nllb-200-distilled-600M"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use half precision for speed
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print(f"NLLB model loaded on {self.device} with optimizations")
        except Exception as e:
            print(f"Error loading NLLB model: {e}")
            return
    
    def translate_text(self, text, source_lang="auto", target_lang="eng_Latn"):
        """Translate text using NLLB"""
        if not self.model or not self.tokenizer:
            print("NLLB model or tokenizer not loaded")
            return
            
        try:
            print(f"Translating: '{text}' from {source_lang} to {target_lang}")
            
            # Language code mapping
            lang_codes = {
                "es": "spa_Latn", "en": "eng_Latn", "fr": "fra_Latn", 
                "de": "deu_Latn", "it": "ita_Latn", "pt": "por_Latn",
                "ru": "rus_Cyrl", "zh": "zho_Hans", "ja": "jpn_Jpan",
                "ko": "kor_Hang", "ar": "arb_Arab", "hi": "hin_Deva"
            }
            
            # Auto-detect source language if needed
            if source_lang == "auto":
                # Simple language detection based on common words
                text_lower = text.lower()
                if any(word in text_lower for word in ["el", "la", "de", "que", "y", "es", "un", "una"]):
                    source_lang = "spa_Latn"
                    print(f"Detected Spanish text: {text}")
                elif any(word in text_lower for word in ["le", "la", "de", "que", "et", "est", "un", "une"]):
                    source_lang = "fra_Latn"
                elif any(word in text_lower for word in ["der", "die", "das", "und", "in", "ist", "ein", "eine"]):
                    source_lang = "deu_Latn"
                elif any(word in text_lower for word in ["il", "la", "di", "che", "e", "è", "un", "una"]):
                    source_lang = "ita_Latn"
                else:
                    source_lang = "eng_Latn"  # Default fallback
                    print(f"Defaulting to English for: {text}")
            
            # Set source language in tokenizer
            self.tokenizer.src_lang = source_lang
            
            # Tokenize and translate
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Use the correct method to get target language token ID
                if hasattr(self.tokenizer, 'lang_code_to_id'):
                    target_token_id = self.tokenizer.lang_code_to_id[target_lang]
                else:
                    # Fallback method for newer tokenizer versions
                    target_token_id = self.tokenizer.convert_tokens_to_ids(f"__{target_lang}__")
                
                translated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=target_token_id,
                    max_length=512,
                    num_beams=2,
                    early_stopping=True
                )
            
            translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            
            if translated_text and translated_text != text:
                print(f"Translation successful: '{text}' -> '{translated_text}'")
                self.translation_ready.emit(text, translated_text, source_lang)
            else:
                print(f"Translation failed or no change: '{text}' -> '{translated_text}'")
                
        except Exception as e:
            print(f"NLLB translation error: {e}")
            # Try fallback translation without forced BOS token
            try:
                self._fallback_translate(text, source_lang, target_lang)
            except Exception as e2:
                print(f"Fallback translation also failed: {e2}")
    
    def _fallback_translate(self, text, source_lang, target_lang):
        """Fallback translation method"""
        try:
            # Simple translation without language-specific tokens
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                translated_tokens = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=2,
                    early_stopping=True
                )
            
            translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            
            if translated_text and translated_text != text:
                self.translation_ready.emit(text, translated_text, source_lang)
                
        except Exception as e:
            print(f"Fallback translation error: {e}")
    
    def stop(self):
        self.running = False
        self.wait()

class OCRThread(QThread):
    """Thread for OCR processing"""
    ocr_ready = Signal(str, str)  # text, language
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.reader = None
        self.current_language = "en"
        
    def run(self):
        # Initialize EasyOCR with proper language combinations
        try:
            # Use proper language combinations for EasyOCR
            supported_langs = ['en', 'es']  # Start with English and Spanish
            self.reader = easyocr.Reader(supported_langs, 
                                       gpu=torch.cuda.is_available(), 
                                       verbose=False,
                                       quantize=True)
            print(f"EasyOCR initialized successfully with languages: {supported_langs}")
        except Exception as e:
            print(f"Error initializing EasyOCR: {e}")
            # Fallback to English only
            try:
                self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
                print("EasyOCR fallback to English only")
            except Exception as e2:
                print(f"EasyOCR fallback failed: {e2}")
                self.reader = None
    
    def process_frame(self, frame):
        """Process frame for OCR with better preprocessing"""
        if not self.reader or not self.running:
            return
            
        try:
            # Try multiple preprocessing approaches
            processed_frames = self.get_multiple_preprocessed_frames(frame)
            
            all_results = []
            for processed_frame in processed_frames:
                # Use better OCR parameters for Spanish text
                results = self.reader.readtext(
                    processed_frame, 
                    paragraph=False,
                    width_ths=0.3,  # Much lower threshold for better detection
                    height_ths=0.3,
                    allowlist='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZáéíóúüñÁÉÍÓÚÜÑ0123456789.,!?¿¡:;()[]{}"\' ',
                    blocklist='',
                    detail=1  # Get detailed results
                )
                all_results.extend(results)
            
            if all_results:
                # Combine all detected text with better filtering
                detected_texts = []
                for (bbox, text, confidence) in all_results:
                    clean_text = text.strip()
                    if (confidence > 0.3 and  # Much lower confidence threshold
                        len(clean_text) > 1 and  # Allow shorter text
                        not self.is_noise_text(clean_text)):
                        detected_texts.append(clean_text)
                
                if detected_texts:
                    combined_text = " ".join(detected_texts)
                    print(f"OCR detected: {combined_text}")  # Debug output
                    self.ocr_ready.emit(combined_text, "ocr")
                    
        except Exception as e:
            print(f"OCR processing error: {e}")
    
    def get_multiple_preprocessed_frames(self, frame):
        """Get multiple preprocessed versions of the frame for better OCR"""
        processed_frames = []
        
        # 1. Original frame (sometimes works better)
        processed_frames.append(frame)
        
        # 2. Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        processed_frames.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
        
        # 3. Gaussian blur + adaptive threshold
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_frames.append(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2RGB))
        
        # 4. Otsu thresholding
        _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_frames.append(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2RGB))
        
        # 5. Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh3 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        processed_frames.append(cv2.cvtColor(thresh3, cv2.COLOR_GRAY2RGB))
        
        # 6. Inverted threshold
        thresh4 = cv2.bitwise_not(thresh1)
        processed_frames.append(cv2.cvtColor(thresh4, cv2.COLOR_GRAY2RGB))
        
        # 7. Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        processed_frames.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))
        
        return processed_frames
    
    def preprocess_frame(self, frame):
        """Preprocess frame for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Apply multiple preprocessing techniques
        # 1. Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 2. Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # 3. Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 4. Convert back to 3-channel for EasyOCR
        processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        
        return processed
    
    def is_noise_text(self, text):
        """Check if text is likely noise"""
        if not text:
            return True
        
        # Check for common OCR noise patterns
        noise_patterns = ['|', '||', '|||', '...', '---', '___', '***', '###', '+++', '///', '\\\\\\']
        if any(pattern in text for pattern in noise_patterns):
            return True
        
        # Check if text is mostly special characters
        special_char_count = sum(1 for c in text if not c.isalnum() and c not in ' .,!?¿¡:;()[]{}"\'')
        if special_char_count > len(text) * 0.5:
            return True
        
        # Check if text is too short and contains only numbers/symbols
        if len(text) <= 2 and not any(c.isalpha() for c in text):
            return True
        
        # Check for repeated characters (common OCR error)
        if len(text) > 3 and len(set(text)) <= 2:
            return True
        
        return False
    
    def stop(self):
        self.running = False
        self.wait()

class RealtimeTranslator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Translator - NLLB + Whisper")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize components
        self.audio_capture = AudioCaptureThread()
        self.whisper_thread = WhisperTranscriptionThread()
        self.nllb_thread = NLLBTranslationThread()
        self.ocr_thread = OCRThread()
        
        # Audio processing
        self.audio_buffer = []
        self.last_audio_time = 0
        self.audio_interval = 3.0  # Process audio every 3 seconds (less frequent)
        
        # OCR processing
        self.frame_count = 0
        self.ocr_processing = False  # Prevent overlapping OCR processing
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # UI
        self.setup_ui()
        self.setup_connections()
        self.start_components()
        
        # Timer for processing
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)  # 50ms = ~20 FPS (reduced for better performance)
        
        # Audio processing timer
        self.audio_timer = QTimer()
        self.audio_timer.timeout.connect(self.process_audio_buffer)
        self.audio_timer.start(3000)  # Process audio every 3 seconds
    
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("Real-time Translator")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 600)
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("border: 2px solid #333;")
        layout.addWidget(self.video_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Audio controls
        audio_group = QGroupBox("Audio Translation")
        audio_layout = QVBoxLayout(audio_group)
        
        self.audio_enabled = QCheckBox("Enable Audio Translation")
        self.audio_enabled.setChecked(True)
        audio_layout.addWidget(self.audio_enabled)
        
        self.audio_status = QLabel("Status: Ready")
        audio_layout.addWidget(self.audio_status)
        
        controls_layout.addWidget(audio_group)
        
        # OCR controls
        ocr_group = QGroupBox("OCR Translation")
        ocr_layout = QVBoxLayout(ocr_group)
        
        self.ocr_enabled = QCheckBox("Enable OCR Translation")
        self.ocr_enabled.setChecked(True)
        ocr_layout.addWidget(self.ocr_enabled)
        
        self.ocr_status = QLabel("Status: Ready")
        ocr_layout.addWidget(self.ocr_status)
        
        controls_layout.addWidget(ocr_group)
        
        # Language controls
        lang_group = QGroupBox("Language Settings")
        lang_layout = QVBoxLayout(lang_group)
        
        lang_layout.addWidget(QLabel("Target Language:"))
        self.target_lang_combo = QComboBox()
        self.target_lang_combo.addItems([
            "English (eng_Latn)", "Spanish (spa_Latn)", "French (fra_Latn)",
            "German (deu_Latn)", "Italian (ita_Latn)", "Portuguese (por_Latn)",
            "Russian (rus_Cyrl)", "Chinese (zho_Hans)", "Japanese (jpn_Jpan)",
            "Korean (kor_Hang)", "Arabic (arb_Arab)", "Hindi (hin_Deva)"
        ])
        self.target_lang_combo.setCurrentText("English (eng_Latn)")
        lang_layout.addWidget(self.target_lang_combo)
        
        controls_layout.addWidget(lang_group)
        
        layout.addLayout(controls_layout)
        
        # Results display
        results_layout = QHBoxLayout()
        
        # Audio results
        audio_results = QGroupBox("Audio Translation Results")
        audio_results_layout = QVBoxLayout(audio_results)
        
        audio_results_layout.addWidget(QLabel("Detected Speech:"))
        self.audio_detected = QTextEdit()
        self.audio_detected.setMaximumHeight(100)
        self.audio_detected.setReadOnly(True)
        audio_results_layout.addWidget(self.audio_detected)
        
        audio_results_layout.addWidget(QLabel("Translated Text:"))
        self.audio_translated = QTextEdit()
        self.audio_translated.setMaximumHeight(100)
        self.audio_translated.setReadOnly(True)
        audio_results_layout.addWidget(self.audio_translated)
        
        results_layout.addWidget(audio_results)
        
        # OCR results
        ocr_results = QGroupBox("OCR Translation Results")
        ocr_results_layout = QVBoxLayout(ocr_results)
        
        ocr_results_layout.addWidget(QLabel("Detected Text:"))
        self.ocr_detected = QTextEdit()
        self.ocr_detected.setMaximumHeight(100)
        self.ocr_detected.setReadOnly(True)
        ocr_results_layout.addWidget(self.ocr_detected)
        
        ocr_results_layout.addWidget(QLabel("Translated Text:"))
        self.ocr_translated = QTextEdit()
        self.ocr_translated.setMaximumHeight(100)
        self.ocr_translated.setReadOnly(True)
        ocr_results_layout.addWidget(self.ocr_translated)
        
        results_layout.addWidget(ocr_results)
        
        layout.addLayout(results_layout)
        
        # Status bar
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
    
    def setup_connections(self):
        """Setup signal connections"""
        self.audio_capture.audio_ready.connect(self.add_audio_data)
        self.whisper_thread.transcription_ready.connect(self.on_transcription_ready)
        self.nllb_thread.translation_ready.connect(self.on_translation_ready)
        self.ocr_thread.ocr_ready.connect(self.on_ocr_ready)
        
        self.audio_enabled.toggled.connect(self.toggle_audio)
        self.ocr_enabled.toggled.connect(self.toggle_ocr)
        self.target_lang_combo.currentTextChanged.connect(self.change_target_language)
    
    def start_components(self):
        """Start all components"""
        # Start threads
        self.whisper_thread.start()
        self.nllb_thread.start()
        self.ocr_thread.start()
        
        # Start audio capture if enabled
        if self.audio_enabled.isChecked():
            self.audio_capture.start()
    
    def add_audio_data(self, audio_data):
        """Add audio data to buffer"""
        if self.audio_enabled.isChecked():
            self.audio_buffer.extend(audio_data)
    
    def process_audio_buffer(self):
        """Process audio buffer with Whisper"""
        if (self.audio_enabled.isChecked() and 
            len(self.audio_buffer) > 16000 and  # At least 1 second of audio
            time.time() - self.last_audio_time > self.audio_interval):
            
            # Process in separate thread to avoid blocking
            threading.Thread(target=self._process_audio_thread, daemon=True).start()
    
    def _process_audio_thread(self):
        """Process audio in separate thread"""
        try:
            self.whisper_thread.add_audio(self.audio_buffer)
            self.whisper_thread.process_audio()
            self.audio_buffer = []
            self.last_audio_time = time.time()
        except Exception as e:
            print(f"Audio processing error: {e}")
    
    def on_transcription_ready(self, text, language):
        """Handle Whisper transcription result"""
        self.audio_detected.setPlainText(f"[{language}] {text}")
        self.audio_status.setText(f"Status: Transcribed ({language})")
        
        # Translate with NLLB in separate thread
        threading.Thread(target=self._translate_text_thread, args=(text, language), daemon=True).start()
    
    def _translate_text_thread(self, text, language):
        """Translate text in separate thread"""
        try:
            self.nllb_thread.translate_text(text, language, self.get_target_language_code())
        except Exception as e:
            print(f"Translation error: {e}")
    
    def _translate_ocr_text_thread(self, text):
        """Translate OCR text in separate thread"""
        try:
            print(f"Starting OCR translation for: {text}")  # Debug output
            self.nllb_thread.translate_text(text, "auto", self.get_target_language_code())
        except Exception as e:
            print(f"OCR translation error: {e}")
    
    def on_translation_ready(self, original, translated, source_lang):
        """Handle NLLB translation result"""
        print(f"Translation ready: '{original}' -> '{translated}' (source: {source_lang})")  # Debug output
        
        if source_lang == "ocr":
            self.ocr_translated.setPlainText(translated)
            self.ocr_status.setText("Status: Translated")
        else:
            self.audio_translated.setPlainText(translated)
            self.audio_status.setText("Status: Translated")
    
    def on_ocr_ready(self, text, language):
        """Handle OCR result"""
        if self.ocr_enabled.isChecked():
            self.ocr_detected.setPlainText(text)
            self.ocr_status.setText("Status: Text Detected")
            print(f"OCR text ready for translation: {text}")  # Debug output
            
            # Translate with NLLB in separate thread
            threading.Thread(target=self._translate_ocr_text_thread, args=(text,), daemon=True).start()
    
    def update_frame(self):
        """Update video frame and process OCR"""
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Display frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))
        
        # Process OCR if enabled (every 3rd frame for better performance)
        if self.ocr_enabled.isChecked() and not self.ocr_processing:
            self.frame_count += 1
            if self.frame_count % 3 == 0:  # Process every 3rd frame for better responsiveness
                self.ocr_processing = True
                # Process in separate thread to avoid blocking
                threading.Thread(target=self._process_ocr_frame, args=(rgb,), daemon=True).start()
    
    def _process_ocr_frame(self, frame):
        """Process OCR frame in separate thread"""
        try:
            self.ocr_thread.process_frame(frame)
        finally:
            self.ocr_processing = False
    
    def get_target_language_code(self):
        """Get target language code from combo box"""
        text = self.target_lang_combo.currentText()
        return text.split('(')[1].split(')')[0]
    
    def toggle_audio(self, enabled):
        """Toggle audio translation"""
        if enabled:
            if not self.audio_capture.isRunning():
                self.audio_capture.start()
            self.audio_status.setText("Status: Recording")
        else:
            if self.audio_capture.isRunning():
                self.audio_capture.stop()
            self.audio_status.setText("Status: Disabled")
    
    def toggle_ocr(self, enabled):
        """Toggle OCR translation"""
        if enabled:
            self.ocr_status.setText("Status: Active")
        else:
            self.ocr_status.setText("Status: Disabled")
    
    def change_target_language(self, language_text):
        """Change target language"""
        self.status_label.setText(f"Target language: {language_text}")
    
    def closeEvent(self, event):
        """Clean up on close"""
        self.audio_capture.stop()
        self.whisper_thread.stop()
        self.nllb_thread.stop()
        self.ocr_thread.stop()
        self.cap.release()
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = RealtimeTranslator()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
