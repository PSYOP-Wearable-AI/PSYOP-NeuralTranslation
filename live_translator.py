import sys, cv2, torch
import easyocr
import numpy as np
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton, QTextEdit, QSlider, QHBoxLayout, QCheckBox, QComboBox
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt, QThread, Signal, QMutex
from transformers import MarianMTModel, MarianTokenizer
import time
import threading

# -------------------------------
# OCR + Translation Worker Thread
# -------------------------------
class OCRWorker(QThread):
    result_ready = Signal(str, str)  # spanish, english

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.mutex = QMutex()
        
        # Language settings
        self.current_language = 'es'
        self.reader = None
        self.paddle_reader = None
        self.active_engine = 'easy'
        self.ocr_engine_preference = 'auto'
        self.initialize_reader()
        
        # Initialize NLLB translation model configuration
        self.nllb_model_name = "facebook/nllb-200-distilled-600M"
        self.nllb_language_codes = {
            'es': 'spa_Latn',
            'fr': 'fra_Latn',
            'de': 'deu_Latn',
            'it': 'ita_Latn',
            'pt': 'por_Latn',
            'ru': 'rus_Cyrl',
            'zh': 'zho_Hans',
            'ja': 'jpn_Jpan',
            'ko': 'kor_Hang',
            'ar': 'arb_Arab'
        }
        self.tokenizer = None
        self.model = None
        self.translation_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not torch.cuda.is_available():
            self.translation_device = torch.device('mps')
        self.initialize_translation_model()
        
        # Performance settings
        self.confidence_threshold = 0.8  # Higher confidence for Spanish only
        self.min_text_length = 4
        self.last_text = ""
        self.similarity_threshold = 0.9
        self.text_history = []
        self.max_history = 5
        
        # Intelligent processing
        self.processing_queue = []
        self.is_processing = False
        self.last_process_time = 0
        self.min_process_interval = 0.5  # Minimum 500ms between processing
        
        # Smart text detection
        self.text_regions = []
        self.region_stability_threshold = 3
        self.stable_regions = {}

    def run(self):
        pass  # worker triggered externally

    def initialize_reader(self):
        """Initialize OCR reader for current language"""
        try:
            easyocr_language = self._map_language_code(self.current_language)
            use_cuda = torch.cuda.is_available()
            self.reader = easyocr.Reader(
                [easyocr_language],
                gpu=use_cuda,
                verbose=False,
                quantize=False,
                model_storage_directory='./models',
                download_enabled=True
            )
        except Exception as e:
            print(f"Error initializing OCR reader: {e}")
            # Fallback to English if language not supported
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    def _map_language_code(self, language_code):
        """Map internal language code to EasyOCR specific code"""
        mapping = {
            'zh': 'ch_sim',
            'ja': 'ja',
            'ko': 'ko',
        }
        return mapping.get(language_code, language_code)

    def _get_allowlist_for_language(self):
        """Return per-language allowlist, if applicable"""
        common_symbols = "0123456789.,!?¿¡:;()[]{}\"' -"
        latin_base = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

        allowlists = {
            'es': latin_base + "áéíóúüñÁÉÍÓÚÜÑ" + common_symbols,
            'fr': latin_base + "àâäéèêëïîôùûüÿçÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ" + common_symbols,
            'de': latin_base + "äöüßÄÖÜ" + common_symbols,
            'it': latin_base + "àèéìíîòóùú" + common_symbols,
            'pt': latin_base + "áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ" + common_symbols,
            'ru': None,
            'zh': None,
            'ja': None,
            'ko': None,
            'ar': None,
        }

        return allowlists.get(self.current_language, latin_base + common_symbols)

    def initialize_translation_model(self):
        """Initialize the shared NLLB translation model"""
        if self.tokenizer is not None and self.model is not None:
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.nllb_model_name)

            model_kwargs = {"low_cpu_mem_usage": True}
            # Use float16 on CUDA for better throughput
            if self.translation_device.type == 'cuda':
                model_kwargs["torch_dtype"] = torch.float16

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.nllb_model_name,
                **model_kwargs,
            )

            self.model.to(self.translation_device)
            if self.translation_device.type != 'cuda':
                # Keep model weights in float32 on CPU/MPS for compatibility
                self.model = self.model.to(torch.float32)

            self.model.eval()

        except Exception as e:
            print(f"Error initializing translation model: {e}")
            self.tokenizer = None
            self.model = None

    def change_language(self, language_code):
        """Change the OCR and translation language"""
        if language_code != self.current_language:
            self.current_language = language_code
            self.initialize_reader()
            self.initialize_translation_model()
            # Clear history when changing languages
            self.text_history = []
            self.last_text = ""

    def calculate_similarity(self, text1, text2):
        """Calculate simple text similarity to avoid reprocessing similar text"""
        if not text1 or not text2:
            return 0.0
        # Simple character-based similarity
        common_chars = sum(1 for c in text1 if c in text2)
        return common_chars / max(len(text1), len(text2))
    
    def is_ghost_text(self, text):
        """Check if text is likely ghost text by comparing with history"""
        if not text:
            return True

        if len(text) < 2:
            return True

        if len(text) == 2 and not any(c.isalpha() for c in text):
            return True
            
        # Check against recent history
        for history_text in self.text_history:
            if self.calculate_similarity(text, history_text) > 0.8:
                return True
                
        # Check for common OCR noise patterns
        noise_patterns = ['|', '||', '|||', '...', '---', '___', '***']
        if any(pattern in text for pattern in noise_patterns):
            return True
            
        return False

    def detect_text_regions(self, frame):
        """Intelligent text region detection using image processing"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding for better text detection
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to connect text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio
        text_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area for text
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.1 < aspect_ratio < 10:  # Reasonable aspect ratio for text
                    text_regions.append((x, y, w, h))
        
        return text_regions

    def process_frame_async(self, frame):
        """Process frame asynchronously to prevent blocking"""
        if not self.running or self.is_processing:
            return
            
        current_time = time.time()
        if current_time - self.last_process_time < self.min_process_interval:
            return
            
        self.is_processing = True
        self.last_process_time = current_time
        
        # Run in separate thread to prevent blocking
        threading.Thread(target=self._process_frame_thread, args=(frame,), daemon=True).start()

    def _process_frame_thread(self, frame):
        """Thread-safe frame processing"""
        try:
            # Intelligent text region detection
            text_regions = self.detect_text_regions(frame)
            
            # Only process if we have stable text regions
            if not self._is_text_stable(text_regions):
                self.is_processing = False
                return
            
            # Resize frame for faster processing
            height, width = frame.shape[:2]
            if width > 800:
                scale = 800 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Use intelligent OCR with region focus - Spanish only
            ocr_kwargs = {
                'paragraph': False,
                'width_ths': 0.7,
                'height_ths': 0.7,
            }

            allowlist = self._get_allowlist_for_language()
            if allowlist:
                ocr_kwargs['allowlist'] = allowlist

            results = self.reader.readtext(frame, **ocr_kwargs)
            
            if not results:
                self.is_processing = False
                return
                
            # Smart filtering
            filtered_results = self._filter_results_intelligently(results)
            
            if not filtered_results:
                self.is_processing = False
                return
                
            paragraphs = self._group_into_paragraphs(filtered_results)
            if not paragraphs:
                self.is_processing = False
                return

            spanish_text = "\n".join(paragraphs)
            
            # Skip if text is too similar to last processed text
            if self.calculate_similarity(spanish_text, self.last_text) > self.similarity_threshold:
                self.is_processing = False
                return
                
            # Update history
            self.text_history.append(spanish_text)
            if len(self.text_history) > self.max_history:
                self.text_history.pop(0)
                
            self.last_text = spanish_text
            
            # Fast translation with caching
            english_text = self._translate_fast(spanish_text)
            if english_text:
                self.result_ready.emit(spanish_text, english_text)
                
        except Exception as e:
            print(f"Processing error: {e}")
        finally:
            self.is_processing = False

    def _is_text_stable(self, text_regions):
        """Check if text regions are stable across frames"""
        if not text_regions:
            return False
            
        # Simple stability check - if regions are similar to previous frame
        if hasattr(self, 'last_regions'):
            if len(text_regions) == len(self.last_regions):
                return True
                
        self.last_regions = text_regions
        return True

    def _filter_results_intelligently(self, results):
        """Intelligent filtering of OCR results"""
        filtered_results = []

        for (bbox, text, confidence) in results:
            clean_text = text.strip()

            # Allow slightly lower confidence for longer spans of text
            adjusted_threshold = self.confidence_threshold
            if len(clean_text) > 40:
                adjusted_threshold = max(0.45, self.confidence_threshold - 0.25)
            elif len(clean_text) > 20:
                adjusted_threshold = max(0.5, self.confidence_threshold - 0.15)

            word_count = len(clean_text.split())
            if word_count >= 2:
                adjusted_threshold = min(adjusted_threshold, max(0.55, self.confidence_threshold - 0.2))
            
            # Reject very short fragments unless they are high confidence words (e.g., "de", "un")
            if len(clean_text) < self.min_text_length:
                if not (len(clean_text) >= 2 and confidence >= max(0.85, adjusted_threshold)):
                    continue

            # Multi-layer filtering
            if (confidence >= adjusted_threshold and 
                not self.is_ghost_text(clean_text) and
                self._is_valid_text(clean_text)):
                filtered_results.append({
                    'text': clean_text,
                    'confidence': confidence,
                    'bbox': bbox,
                })

        return filtered_results

    def _group_into_paragraphs(self, filtered_results):
        """Cluster OCR lines into paragraphs preserving reading order"""
        if not filtered_results:
            return []

        lines = []
        for entry in filtered_results:
            bbox = entry['bbox']
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            top = min(ys)
            bottom = max(ys)
            left = min(xs)
            right = max(xs)
            height = max(bottom - top, 1)

            lines.append({
                'text': entry['text'],
                'confidence': entry['confidence'],
                'top': top,
                'bottom': bottom,
                'left': left,
                'right': right,
                'height': height,
            })

        if not lines:
            return []

        lines.sort(key=lambda item: (item['top'], item['left']))

        heights = [item['height'] for item in lines]
        median_height = float(np.median(heights)) if heights else 20.0
        row_threshold = max(10.0, median_height * 0.6)

        rows = []
        for line in lines:
            placed = False
            for row in rows:
                if abs(line['top'] - row['top']) <= row_threshold:
                    row['items'].append(line)
                    row['top_values'].append(line['top'])
                    row['bottom_values'].append(line['bottom'])
                    placed = True
                    break
            if not placed:
                rows.append({
                    'items': [line],
                    'top_values': [line['top']],
                    'bottom_values': [line['bottom']],
                    'top': line['top'],
                })

        aggregated_rows = []
        for row in rows:
            items = sorted(row['items'], key=lambda item: item['left'])
            text = " ".join(item['text'] for item in items)
            top = min(row['top_values'])
            bottom = max(row['bottom_values'])
            aggregated_rows.append({
                'text': text,
                'top': top,
                'bottom': bottom,
                'height': max(bottom - top, 1),
            })

        if not aggregated_rows:
            return []

        aggregated_rows.sort(key=lambda row: row['top'])

        row_heights = [row['height'] for row in aggregated_rows]
        median_row_height = float(np.median(row_heights)) if row_heights else 20.0
        paragraph_gap_threshold = max(25.0, median_row_height * 1.3)

        paragraphs = []
        current_paragraph = []
        last_bottom = None

        for row in aggregated_rows:
            if not current_paragraph:
                current_paragraph = [row]
            else:
                gap = row['top'] - last_bottom if last_bottom is not None else 0
                if gap > paragraph_gap_threshold:
                    paragraphs.append(current_paragraph)
                    current_paragraph = [row]
                else:
                    current_paragraph.append(row)
            last_bottom = row['bottom']

        if current_paragraph:
            paragraphs.append(current_paragraph)

        paragraph_texts = []
        for paragraph in paragraphs:
            paragraph.sort(key=lambda row: row['top'])
            joined = " ".join(row['text'] for row in paragraph)
            paragraph_texts.append(joined)

        return paragraph_texts

    def _is_valid_text(self, text):
        """Additional text validation based on current language"""
        # Check for reasonable text patterns
        if text.count(' ') > len(text) * 0.5:  # Too many spaces
            return False

        # Require at least a couple of readable characters
        letters = [c for c in text if c.isalpha()]
        if len(letters) < 2 and not any(c.isalnum() for c in text):
            return False

        # Language-specific validation
        if self.current_language == 'es':
            return self._is_spanish_text(text)
        elif self.current_language == 'fr':
            return self._is_french_text(text)
        elif self.current_language == 'de':
            return self._is_german_text(text)
        elif self.current_language == 'it':
            return self._is_italian_text(text)
        elif self.current_language == 'pt':
            return self._is_portuguese_text(text)
        elif self.current_language == 'ru':
            return self._is_russian_text(text)
        elif self.current_language == 'zh':
            return self._is_chinese_text(text)
        elif self.current_language == 'ja':
            return self._is_japanese_text(text)
        elif self.current_language == 'ko':
            return self._is_korean_text(text)
        elif self.current_language == 'ar':
            return self._is_arabic_text(text)
        else:
            return True  # Default validation for other languages

    def _has_language_character_support(self, text, charset, min_ratio=0.4, min_matches=2):
        """Check that text contains enough characters from the target language"""
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return False

        matches = sum(1 for c in letters if c in charset)
        if matches < min_matches:
            return False

        return matches / len(letters) >= min_ratio

    def _is_spanish_text(self, text):
        """Validate Spanish text"""
        spanish_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZáéíóúüñÁÉÍÓÚÜÑ')
        return self._has_language_character_support(text, spanish_chars)

    def _is_french_text(self, text):
        """Validate French text"""
        french_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZàâäéèêëïîôùûüÿçÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ')
        return self._has_language_character_support(text, french_chars)

    def _is_german_text(self, text):
        """Validate German text"""
        german_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüßÄÖÜ')
        return self._has_language_character_support(text, german_chars)

    def _is_italian_text(self, text):
        """Validate Italian text"""
        italian_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZàèéìíîòóùú')
        return self._has_language_character_support(text, italian_chars)

    def _is_portuguese_text(self, text):
        """Validate Portuguese text"""
        portuguese_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZáàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ')
        return self._has_language_character_support(text, portuguese_chars)

    def _is_russian_text(self, text):
        """Validate Russian text"""
        russian_chars = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
        return self._has_language_character_support(text, russian_chars, min_ratio=0.6, min_matches=1)

    def _is_chinese_text(self, text):
        """Validate Chinese text"""
        chinese_chars = set('的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当')
        return self._has_language_character_support(text, chinese_chars, min_ratio=0.3, min_matches=1)

    def _is_japanese_text(self, text):
        """Validate Japanese text"""
        japanese_chars = set('あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポ')
        return self._has_language_character_support(text, japanese_chars, min_ratio=0.3, min_matches=1)

    def _is_korean_text(self, text):
        """Validate Korean text"""
        korean_chars = set('가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후그느드르므브스으즈츠크트프흐기니디리미비시이지치키티피히')
        return self._has_language_character_support(text, korean_chars, min_ratio=0.3, min_matches=1)

    def _is_arabic_text(self, text):
        """Validate Arabic text"""
        arabic_chars = set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
        return self._has_language_character_support(text, arabic_chars, min_ratio=0.3, min_matches=1)

    def _translate_fast(self, text):
        """Fast translation with error handling and better accuracy"""
        try:
            if self.tokenizer is None or self.model is None:
                return None

            # Limit text length for speed while allowing multi-line context
            if len(text) > 400:
                text = text[:400]

            source_code = self.nllb_language_codes.get(self.current_language, 'eng_Latn')
            target_code = 'eng_Latn'

            self.tokenizer.src_lang = source_code

            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            tokens = {k: v.to(self.translation_device) for k, v in tokens.items()}

            generation_kwargs = {
                "max_length": 256,
                "num_beams": 3,
                "early_stopping": True,
                "repetition_penalty": 1.05,
            }

            if hasattr(self.tokenizer, "lang_code_to_id") and target_code in self.tokenizer.lang_code_to_id:
                generation_kwargs["forced_bos_token_id"] = self.tokenizer.lang_code_to_id[target_code]

            with torch.no_grad():
                translated = self.model.generate(**tokens, **generation_kwargs)

            english_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            
            # Clean up the translation
            english_text = english_text.strip()
            if english_text and english_text != text:  # Make sure it's actually translated
                return english_text
            return None
        except Exception as e:
            print(f"Translation error: {e}")
            return None

    def process_frame(self, frame):
        """Main entry point for frame processing"""
        self.process_frame_async(frame)

    def update_confidence_threshold(self, value):
        self.confidence_threshold = value / 100.0

# -------------------------------
# Main Window
# -------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Field Translator - Optimized")
        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 600)
        self.video_label.setScaledContents(True)  # Scale to fit the label

        # HUD configuration for wearable display
        self.hud_mode = False
        self.overlay_spanish = ""
        self.overlay_english = ""
        self.on_external_display = False

        self.text_spanish = QTextEdit()
        self.text_spanish.setReadOnly(True)
        self.text_spanish.setMaximumHeight(100)
        self.text_english = QTextEdit()
        self.text_english.setReadOnly(True)
        self.text_english.setMaximumHeight(100)

        self.btn_toggle = QPushButton("Pause OCR")
        self.btn_toggle.clicked.connect(self.toggle_ocr)

        self.btn_hud = QPushButton("Enable HUD Overlay")
        self.btn_hud.clicked.connect(self.toggle_hud_mode)

        self.btn_external_display = QPushButton("Send to External Display")
        self.btn_external_display.clicked.connect(self.toggle_external_display)

        # Performance controls
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(10)
        self.confidence_slider.setMaximum(90)
        self.confidence_slider.setValue(80)  # Higher default for Spanish only
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        
        self.confidence_label = QLabel("Confidence: 80%")
        
        # Frame skip controls
        self.frame_skip_slider = QSlider(Qt.Horizontal)
        self.frame_skip_slider.setMinimum(1)
        self.frame_skip_slider.setMaximum(10)
        self.frame_skip_slider.setValue(5)  # Higher default for better performance
        self.frame_skip_slider.valueChanged.connect(self.update_frame_skip)
        
        self.frame_skip_label = QLabel("Process every 5 frames")
        
        # Language selection
        self.language_combo = QComboBox()
        self.language_combo.addItems([
            "Spanish (es)", "French (fr)", "German (de)", "Italian (it)", 
            "Portuguese (pt)", "Russian (ru)", "Chinese (zh)", "Japanese (ja)", 
            "Korean (ko)", "Arabic (ar)"
        ])
        self.language_combo.setCurrentText("Spanish (es)")
        self.language_combo.currentTextChanged.connect(self.change_language)
        
        # Intelligent processing controls
        self.smart_processing_checkbox = QCheckBox("Smart Processing")
        self.smart_processing_checkbox.setChecked(True)
        self.smart_processing_checkbox.stateChanged.connect(self.toggle_smart_processing)
        
        self.region_detection_checkbox = QCheckBox("Region Detection")
        self.region_detection_checkbox.setChecked(True)
        self.region_detection_checkbox.stateChanged.connect(self.toggle_region_detection)
        
        # Performance info
        self.performance_label = QLabel("FPS: -- | Processing: -- | Status: Ready")
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        
        # Performance controls layout
        perf_layout = QHBoxLayout()
        perf_layout.addWidget(QLabel("Confidence:"))
        perf_layout.addWidget(self.confidence_slider)
        perf_layout.addWidget(self.confidence_label)
        layout.addLayout(perf_layout)
        
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Frame Skip:"))
        frame_layout.addWidget(self.frame_skip_slider)
        frame_layout.addWidget(self.frame_skip_label)
        layout.addLayout(frame_layout)
        
        # Language selection
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Language:"))
        lang_layout.addWidget(self.language_combo)
        layout.addLayout(lang_layout)
        
        # Smart processing controls
        smart_layout = QHBoxLayout()
        smart_layout.addWidget(self.smart_processing_checkbox)
        smart_layout.addWidget(self.region_detection_checkbox)
        layout.addLayout(smart_layout)
        
        self.detected_label = QLabel("Detected Text:")
        layout.addWidget(self.detected_label)
        layout.addWidget(self.text_spanish)
        self.translated_label = QLabel("Translated English:")
        layout.addWidget(self.translated_label)
        layout.addWidget(self.text_english)
        button_row = QHBoxLayout()
        button_row.addWidget(self.btn_toggle)
        button_row.addWidget(self.btn_hud)
        button_row.addWidget(self.btn_external_display)
        layout.addLayout(button_row)
        layout.addWidget(self.performance_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        # Set camera resolution for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms = ~33 FPS

        # Worker
        self.ocr_worker = OCRWorker()
        self.ocr_worker.result_ready.connect(self.display_result)
        self.ocr_active = True
        
        # Performance tracking
        self.frame_count = 0
        self.process_count = 0
        self.frame_skip = 5  # Updated default
        self.last_fps_time = 0
        self.fps_counter = 0
        self.processing_status = "Ready"

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
            
        self.frame_count += 1
        self.fps_counter += 1
        
        # Convert and display frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        if self.hud_mode:
            hud_frame = rgb.copy()
            self._draw_hud_overlay(hud_frame)
            frame_to_show = hud_frame
        else:
            frame_to_show = rgb

        qimg = QImage(frame_to_show.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

        # Process only every N frames when OCR is active
        if self.ocr_active and self.frame_count % self.frame_skip == 0:
            self.ocr_worker.process_frame(rgb)
            self.process_count += 1
            self.processing_status = "Processing"
        
        # Update FPS counter every second
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
            
            # Update status
            if not self.ocr_worker.is_processing:
                self.processing_status = "Ready"
            
            self.performance_label.setText(f"FPS: {fps} | Processing: {self.process_count} frames | Status: {self.processing_status}")
            self.process_count = 0

    def display_result(self, spanish, english):
        self.text_spanish.setPlainText(spanish)
        self.text_english.setPlainText(english)
        self.overlay_spanish = spanish
        self.overlay_english = english

    def toggle_ocr(self):
        self.ocr_active = not self.ocr_active
        self.btn_toggle.setText("Resume OCR" if not self.ocr_active else "Pause OCR")

    def update_confidence(self, value):
        self.ocr_worker.update_confidence_threshold(value)
        self.confidence_label.setText(f"Confidence: {value}%")

    def update_frame_skip(self, value):
        self.frame_skip = value
        self.frame_skip_label.setText(f"Process every {value} frames")

    def toggle_smart_processing(self, state):
        """Toggle intelligent processing features"""
        if state == Qt.Checked:
            self.ocr_worker.min_process_interval = 0.3  # Faster processing
            self.ocr_worker.similarity_threshold = 0.9
        else:
            self.ocr_worker.min_process_interval = 0.5  # Slower processing
            self.ocr_worker.similarity_threshold = 0.8

    def toggle_region_detection(self, state):
        """Toggle region-based text detection"""
        if state == Qt.Checked:
            self.ocr_worker.region_stability_threshold = 3
        else:
            self.ocr_worker.region_stability_threshold = 1

    def toggle_hud_mode(self):
        self.hud_mode = not self.hud_mode
        self.btn_hud.setText("Disable HUD Overlay" if self.hud_mode else "Enable HUD Overlay")

        # Hide text panels when HUD is active for cleaner glasses experience
        self.text_spanish.setVisible(not self.hud_mode)
        self.text_english.setVisible(not self.hud_mode)
        self.detected_label.setVisible(not self.hud_mode)
        self.translated_label.setVisible(not self.hud_mode)

    def toggle_external_display(self):
        self.on_external_display = not self.on_external_display

        if self.on_external_display:
            screens = QApplication.screens()
            if len(screens) > 1:
                target = screens[-1]
                window_handle = self.windowHandle()
                if window_handle:
                    window_handle.setScreen(target)
                self.showFullScreen()
                self.btn_external_display.setText("Return to Main Display")
            else:
                self.on_external_display = False
                self.btn_external_display.setText("Send to External Display")
        else:
            self.showNormal()
            self.resize(900, 900)
            self.btn_external_display.setText("Send to External Display")

    def _draw_hud_overlay(self, frame):
        """Draw translated text directly onto the video frame for HUD mode."""
        h, w, _ = frame.shape
        y_start = int(h * 0.7)
        spacing = int(h * 0.08)
        font_scale = max(w, h) / 900
        font_scale = max(1.0, min(font_scale, 1.8))
        line_thickness = max(2, int(font_scale * 2))

        hud_lines = []
        if self.overlay_spanish:
            hud_lines.append((f"ES: {self.overlay_spanish}", (255, 255, 255)))
        if self.overlay_english:
            hud_lines.append((f"EN: {self.overlay_english}", (0, 255, 0)))

        if not hud_lines:
            return

        for idx, (text, color) in enumerate(hud_lines):
            y = min(h - 20, y_start + idx * spacing)

            # Background rectangle for readability
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)
            box_w, box_h = text_size
            padding = 10
            x = 30
            cv2.rectangle(
                frame,
                (x - padding, y - box_h - padding),
                (x + box_w + padding, y + padding // 2),
                (0, 0, 0),
                cv2.FILLED,
            )

            # Outline for better contrast
            cv2.putText(
                frame,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                line_thickness + 2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                line_thickness,
                cv2.LINE_AA,
            )

    def change_language(self, language_text):
        """Change the OCR and translation language"""
        # Extract language code from text like "Spanish (es)"
        language_code = language_text.split('(')[1].split(')')[0]
        self.ocr_worker.change_language(language_code)
        
        # Update window title
        language_name = language_text.split('(')[0].strip()
        self.setWindowTitle(f"Field Translator - {language_name}")

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
