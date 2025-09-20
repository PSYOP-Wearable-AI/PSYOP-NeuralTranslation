#!/usr/bin/env python3
import time
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

from picamera2 import Picamera2

from argostranslate import package, translate
from deep_translator import GoogleTranslator

def ensure_argos_model(src_code="auto", dst_code="en"):
    try:
        installed = translate.get_installed_languages()
        src_lang = next((l for l in installed if l.code == src_code), None)
        dst_lang = next((l for l in installed if l.code == dst_code), None)
        if src_lang and dst_lang and any(
            t.from_language.code == src_code and t.to_language.code == dst_code
            for t in src_lang.translations
        ):
            return True

        if src_code == "auto":
            return False

        available = package.get_available_packages()
        candidates = [p for p in available if p.from_code == src_code and p.to_code == dst_code]
        if candidates:
            pkg = sorted(candidates, key=lambda p: p.size)[0]
            download_path = pkg.download()
            package.install_from_path(download_path)
            return True
    except Exception:
        pass
    return False

def argos_translate(text, src_code, dst_code="en"):
    installed = translate.get_installed_languages()
    src_lang = next((l for l in installed if l.code == src_code), None)
    dst_lang = next((l for l in installed if l.code == dst_code), None)
    if not (src_lang and dst_lang):
        raise RuntimeError("Argos languages not installed")
    translator = next((t for t in src_lang.translations if t.to_language.code == dst_code), None)
    if not translator:
        raise RuntimeError("Argos translator pair missing")
    return translator.translate(text)

def online_translate(text):
    return GoogleTranslator(source="auto", target="en").translate(text)

def capture_frame():
    picam = Picamera2()
    config = picam.create_still_configuration({"size": (1920, 1080)})
    picam.configure(config)
    picam.start()
    time.sleep(0.5)  # warm up
    frame = picam.capture_array()
    picam.stop()
    return frame

def ocr_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(th, config=config)
    return text.strip()

def main():
    print("Capturing image from camera...")
    frame = capture_frame()
    print("Running OCR...")
    text = ocr_from_frame(frame)

    if not text:
        print("\nNo text detected. Try more light, closer framing, or steadier hold.\n")
        return

    print("\n==== OCR TEXT (raw) ====\n")
    print(text)

    try:
        src = detect(text)
    except Exception:
        src = "auto"

    print(f"\nDetected language: {src}")

    translated = None
    used = None

    if src != "en" and ensure_argos_model(src_code=src, dst_code="en"):
        try:
            translated = argos_translate(text, src_code=src, dst_code="en")
            used = "Argos (offline)"
        except Exception:
            translated = None

    if translated is None and src != "en":
        try:
            translated = online_translate(text)
            used = "Google (online)"
        except Exception as e:
            print("\nTranslation failed (offline and online). Error:", e)
            return

    if src == "en":
        translated = text
        used = "N/A"

    print("\n==== TRANSLATION → EN ====\n")
    print(translated)
    print(f"\nTranslator used: {used}")

    # Save a preview image with captioned translation (optional)
    overlay = frame.copy()
    h, w = overlay.shape[:2]
    bar_h = min(200, h // 3)
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    snippet = translated.replace("\n", " ")
    snippet = (snippet[:200] + "…") if len(snippet) > 200 else snippet
    cv2.putText(overlay, snippet, (20, h - bar_h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    out_path = Path("ocr_translation_overlay.jpg")
    cv2.imwrite(str(out_path), overlay)
    print(f"\nSaved preview image with overlay: {out_path.resolve()}\n")

if __name__ == "__main__":
    main()