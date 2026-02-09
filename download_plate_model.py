"""
دانلود مدل تشخیص پلاک فارسی (plateYolo.pt) از پروژه persian-license-plate-recognition.
این مدل برای تشخیص ناحیه پلاک در تصویر خودرو استفاده می‌شود.

اجرا در CMD:
    python download_plate_model.py
"""
import os
import sys
import urllib.request

# آدرس مدل در مخزن GitHub (پروژه PLPR)
PLATE_MODEL_URL = "https://github.com/truthofmatthew/persian-license-plate-recognition/raw/main/model/plateYolo.pt"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "plateYolo.pt")


def download_with_progress(url, dest_path):
    def report(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        if downloaded <= total_size:
            pct = min(100, 100 * downloaded / total_size)
            mb = total_size / (1024 * 1024)
            print(f"\r  دانلود: {pct:.0f}% ({downloaded/(1024*1024):.1f} / {mb:.1f} MB)", end="")
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=report)
        print()
        return True
    except Exception as e:
        print(f"\n  خطا: {e}")
        return False


def main():
    if os.path.isfile(MODEL_PATH):
        print(f"[+] فایل مدل از قبل وجود دارد: {MODEL_PATH}")
        return 0
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("[*] در حال دانلود مدل تشخیص پلاک (plateYolo.pt) ...")
    print("    منبع: https://github.com/truthofmatthew/persian-license-plate-recognition")
    if not download_with_progress(PLATE_MODEL_URL, MODEL_PATH):
        print("[!] دانلود خودکار ممکن است به دلیل محدودیت حجم در GitHub ناموفق باشد.")
        print("    لطفاً به صورت دستی انجام دهید:")
        print("    1. به آدرس زیر بروید:")
        print("       https://github.com/truthofmatthew/persian-license-plate-recognition/tree/main/model")
        print("    2. فایل plateYolo.pt را دانلود کنید.")
        print(f"    3. آن را در این مسیر قرار دهید: {MODEL_PATH}")
        return 1
    print(f"[+] مدل با موفقیت ذخیره شد: {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
