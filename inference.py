"""Простой скрипт для запуска workflow Roboflow и визуализации результата.

Перед запуском:
  1. Вставьте свои API_URL и API_KEY.
  2. Укажите путь к изображению в переменной IMAGE_PATH.
  3. Установите зависимости: pip install -r requirements.txt
  4. Запустите: python inference.py
"""

import base64
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

# ==== НАСТРОЙКИ ===========================================================

# URL вашего workflow в Roboflow (скопировать из вкладки "Integrate")
API_URL = "https://serverless.roboflow.com/YOUR_WORKSPACE/workflows/YOUR_WORKFLOW"

# Ваш API‑ключ Roboflow
API_KEY = "YOUR_API_KEY_HERE"

# Путь к изображению с китом
IMAGE_PATH = "examples/example_whale.jpg"

# Папка для сохранения результатов
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ==== ЛОГИКА ВЫЗОВА WORKFLOW ============================================

def run_workflow(image_path: str):
    """Отправляем изображение в workflow и получаем JSON‑ответ."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "api_key": API_KEY,
        "inputs": {
            "image": {"type": "base64", "value": b64}
        }
    }

    resp = requests.post(API_URL, json=payload)
    resp.raise_for_status()
    return resp.json()

def get_best_prediction(result: dict):
    """Выбираем предсказание с максимальной уверенностью."""
    preds = []

    def collect(node):
        if isinstance(node, dict):
            if "class" in node and "confidence" in node:
                preds.append(node)
            for v in node.values():
                collect(v)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(result)
    if not preds:
        return None
    return max(preds, key=lambda p: p.get("confidence", 0) or 0)

def draw_box_with_label(image: Image.Image, pred: dict) -> Image.Image:
    """Рисуем красную рамку и подпись класса внутри рамки."""
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)

    cls = pred.get("class", "whale")
    conf = pred.get("confidence", 0.0)

    x, y = pred.get("x"), pred.get("y")
    w, h = pred.get("width"), pred.get("height")
    if None in (x, y, w, h):
        return img

    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

    label = f"{cls} ({conf:.2f})"
    try:
        font_size = max(24, img.width // 20)
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    try:
        bbox = font.getbbox(label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textsize(label, font=font)

    pad_x, pad_y = 6, 4
    bg_w, bg_h = tw + 2 * pad_x, th + 2 * pad_y
    bg_x1, bg_y1 = x1, max(0, y1)
    bg_x2, bg_y2 = bg_x1 + bg_w, bg_y1 + bg_h

    # полупрозрачная чёрная плашка
    img_rgba = img.convert("RGBA")
    bg = Image.new("RGBA", (bg_w, bg_h), (0, 0, 0, 190))
    img_rgba.paste(bg, (bg_x1, bg_y1), bg)

    draw = ImageDraw.Draw(img_rgba)
    draw.text((bg_x1 + pad_x, bg_y1 + pad_y), label, fill="white", font=font)

    return img_rgba.convert("RGB")

def main():
    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    img = Image.open(image_path)
    print(f"Отправляю изображение: {image_path}")

    result = run_workflow(str(image_path))
    best = get_best_prediction(result)
    if best is None:
        print("Модель не нашла кита на изображении.")
        return

    out_img = draw_box_with_label(img, best)
    out_path = RESULTS_DIR / f"{image_path.stem}_vis.png"
    out_img.save(out_path)
    print("Результат сохранён в:", out_path)

if __name__ == "__main__":
    main()