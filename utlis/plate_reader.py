"""
ماژول تشخیص و خواندن پلاک ایرانی.
برای تشخیص پلاک از مدل YOLOv5 مخصوص پلاک فارسی (plateYolo.pt) استفاده می‌شود
در صورت نبود فایل مدل، از روش پردازش تصویر (رنگ/شکل) به عنوان fallback استفاده می‌شود.
منبع مدل: https://github.com/truthofmatthew/persian-license-plate-recognition
"""

import os
import cv2
import numpy as np
import re
import time

# مسیر ریشه پروژه (یک سطح بالاتر از utlis)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# مدل تشخیص پلاک YOLOv5 (پلاک فارسی) - از repo persian-license-plate-recognition
_PLATE_YOLOV5_PATH = os.path.join(_PROJECT_ROOT, "model", "plateYolo.pt")
# پوشه ذخیره مدل‌های EasyOCR در پروژه
_EASYOCR_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models", "easyocr")


class IranianPlateReader:
    """کلاس اصلی برای تشخیص و خواندن پلاک ایرانی"""
    
    # حروف فارسی مجاز در پلاک ایرانی
    PERSIAN_LETTERS = {
        'الف': 'الف', 'ب': 'ب', 'پ': 'پ', 'ت': 'ت', 'ث': 'ث',
        'ج': 'ج', 'چ': 'چ', 'ح': 'ح', 'خ': 'خ', 'د': 'د',
        'ذ': 'ذ', 'ر': 'ر', 'ز': 'ز', 'ژ': 'ژ', 'س': 'س',
        'ش': 'ش', 'ص': 'ص', 'ض': 'ض', 'ط': 'ط', 'ظ': 'ظ',
        'ع': 'ع', 'غ': 'غ', 'ف': 'ف', 'ق': 'ق', 'ک': 'ک',
        'گ': 'گ', 'ل': 'ل', 'م': 'م', 'ن': 'ن', 'و': 'و',
        'ه': 'ه', 'ی': 'ی',
    }
    
    def __init__(self, plate_model_path=None, confidence=0.25):
        """
        مقداردهی اولیه.
        اگر مدل plateYolo.pt (YOLOv5 پلاک فارسی) در مسیر model/plateYolo.pt باشد، از آن استفاده می‌شود؛
        در غیر این صورت از روش تشخیص بر اساس رنگ/شکل استفاده می‌شود.
        
        Args:
            plate_model_path: مسیر فایل مدل تشخیص پلاک (plateYolo.pt). پیش‌فرض: model/plateYolo.pt
            confidence: حداقل اطمینان برای تشخیص پلاک (۰.۲۵ برای پلاک‌های کوچک مناسب است)
        """
        self.confidence = confidence
        self.ocr_reader = None
        self._ocr_failed = False
        self._plate_model = None  # مدل YOLOv5 برای تشخیص پلاک
        
        path = plate_model_path or _PLATE_YOLOV5_PATH
        if not os.path.isabs(path):
            path = os.path.join(_PROJECT_ROOT, path) if not os.path.isfile(path) else path
        if os.path.isfile(path):
            try:
                import torch
                print(f"[*] در حال بارگذاری مدل تشخیص پلاک YOLOv5 از: {path}")
                self._plate_model = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=path,
                    force_reload=False,
                    trust_repo=True,
                )
                self._plate_model.conf = self.confidence
                print("[+] مدل تشخیص پلاک (YOLOv5) آماده است!")
            except Exception as e:
                print(f"[!] خطا در بارگذاری مدل پلاک: {e}. از روش تشخیص با رنگ/شکل استفاده می‌شود.")
                self._plate_model = None
        else:
            print("[!] فایل مدل پلاک (plateYolo.pt) یافت نشد. از روش رنگ/شکل استفاده می‌شود.")
            print("    برای دقت بالاتر، مدل را از این آدرس دریافت کنید:")
            print("    https://github.com/truthofmatthew/persian-license-plate-recognition")
            print("    و در پوشه model با نام plateYolo.pt ذخیره کنید.")
    
    @staticmethod
    def _easyocr_models_exist():
        """بررسی وجود فایل‌های مدل EasyOCR در پوشهٔ پروژه تا دانلود مجدد نشود."""
        if not os.path.isdir(_EASYOCR_MODEL_DIR):
            return False
        # EasyOCR حداقل فایل تشخیص (craft) و مدل‌های زبان را ذخیره می‌کند
        for name in os.listdir(_EASYOCR_MODEL_DIR):
            if name.endswith(".pth") or name.endswith(".zip"):
                return True
        return False

    def _get_ocr_reader(self):
        """دریافت یا ایجاد OCR reader (lazy loading). فقط در صورت نبود مدل دانلود می‌شود."""
        if self._ocr_failed:
            return None
        if self.ocr_reader is None:
            try:
                import easyocr
            except ImportError:
                print("[!] خطا: کتابخانه easyocr نصب نیست. pip install easyocr")
                self._ocr_failed = True
                return None
            os.makedirs(_EASYOCR_MODEL_DIR, exist_ok=True)
            # اگر مدل‌ها قبلاً در پوشهٔ پروژه ذخیره شده‌اند، دانلود مجدد نکن
            use_local_only = self._easyocr_models_exist()
            max_attempts = 2 if use_local_only else 3
            for attempt in range(1, max_attempts + 1):
                try:
                    if use_local_only:
                        print("[*] در حال بارگذاری مدل OCR از پوشهٔ محلی (بدون دانلود)...")
                    else:
                        print(f"[*] در حال بارگذاری مدل OCR فارسی (تلاش {attempt}/{max_attempts})...")
                    if attempt > 1:
                        time.sleep(3)
                    self.ocr_reader = easyocr.Reader(
                        ["fa", "en"],
                        gpu=False,
                        model_storage_directory=_EASYOCR_MODEL_DIR,
                        download_enabled=not use_local_only,
                    )
                    print("[+] مدل OCR آماده است!")
                    return self.ocr_reader
                except Exception as e:
                    err_msg = str(e).lower()
                    if use_local_only:
                        # مدل محلی ناقص بود؛ یک بار با دانلود تلاش کن
                        use_local_only = False
                        self.ocr_reader = None
                        continue
                    if "retrieval incomplete" in err_msg or "urlopen" in err_msg or "got only" in err_msg:
                        print(f"[!] دانلود ناقص بود. پاک‌سازی و تلاش مجدد...")
                        self._remove_incomplete_easyocr_models()
                        self.ocr_reader = None
                    else:
                        print(f"[!] خطا در بارگذاری OCR: {e}")
                        self._ocr_failed = True
                        return None
            print("[!] مدل OCR بارگذاری نشد. فقط تشخیص پلاک (بدون خواندن متن) انجام می‌شود.")
            self._ocr_failed = True
        return self.ocr_reader
    
    @staticmethod
    def _remove_incomplete_easyocr_models():
        """پاک کردن فایل‌های ناقص مدل EasyOCR تا در تلاش بعدی دوباره دانلود شوند."""
        if not os.path.isdir(_EASYOCR_MODEL_DIR):
            return
        try:
            for name in os.listdir(_EASYOCR_MODEL_DIR):
                path = os.path.join(_EASYOCR_MODEL_DIR, name)
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    for sub in os.listdir(path):
                        os.remove(os.path.join(path, sub))
                    os.rmdir(path)
        except OSError:
            pass
    
    def detect_plate_region(self, vehicle_image, vehicle_bbox=None):
        """
        تشخیص ناحیه پلاک در تصویر برش‌خوردهٔ خودرو.
        اگر مدل YOLOv5 پلاک (plateYolo.pt) بارگذاری شده باشد از آن استفاده می‌شود؛
        در غیر این صورت از روش رنگ/شکل استفاده می‌شود.
        """
        plates = []
        if self._plate_model is not None:
            plates = self._detect_plate_by_yolov5(vehicle_image)
        if not plates:
            plates = self._detect_plate_by_color(vehicle_image)
        return plates
    
    def _detect_plate_by_yolov5(self, image):
        """
        تشخیص پلاک با مدل YOLOv5 آموزش‌دیده برای پلاک فارسی (plateYolo.pt).
        مدل از پروژه persian-license-plate-recognition است.
        """
        if image is None or image.size == 0:
            return []
        try:
            # YOLOv5 از torch.hub تصویر RGB می‌خواهد
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self._plate_model(img_rgb)
            # خروجی به صورت pandas با ستون‌های xmin, ymin, xmax, ymax, confidence
            df = results.pandas().xyxy[0]
            plates = []
            for _, row in df.iterrows():
                conf = float(row.get("confidence", 0))
                if conf < self.confidence:
                    continue
                x1 = int(row["xmin"])
                y1 = int(row["ymin"])
                x2 = int(row["xmax"])
                y2 = int(row["ymax"])
                h_img, w_img = image.shape[:2]
                # محدود کردن به محدوده تصویر و کمی حاشیه
                x1 = max(0, x1 - 2)
                y1 = max(0, y1 - 2)
                x2 = min(w_img, x2 + 2)
                y2 = min(h_img, y2 + 2)
                if x2 <= x1 or y2 <= y1:
                    continue
                area = (x2 - x1) * (y2 - y1)
                plates.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "area": area,
                })
            return plates
        except Exception as e:
            print(f"[!] خطا در تشخیص پلاک با YOLOv5: {e}")
            return []
    
    def _detect_plate_by_color(self, image):
        """
        تشخیص پلاک بر اساس رنگ و شکل:
        - تمرکز روی یک‌سوم پایین تصویر (محل معمول پلاک)
        - مستطیل با نسبت ابعاد پلاک ایرانی؛ پلاک می‌تواند کمی کج باشد (minAreaRect)
        - رنگ سفید یا زرد (پلاک سفید/زرد)
        """
        h_img, w_img = image.shape[:2]
        # ناحیه جستجو: نیمهٔ پایین تصویر (پلاک اکثراً پایین است)، ترجیحاً یک‌سوم پایین
        search_y_start = int(h_img * 0.45)  # از ۴۵٪ ارتفاع به پایین
        search_roi = image[search_y_start:, :]
        h_roi, w_roi = search_roi.shape[:2]
        
        # ترکیب چند روش برای ماسک بهتر
        gray = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(search_roi, cv2.COLOR_BGR2HSV)
        
        # ماسک سفید/خاکستری روشن (پلاک سفید)
        _, white_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = cv2.inRange(gray, 180, 255)
        
        # ماسک HSV برای سفید (اشباع کم، روشنایی بالا)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 55, 255])
        mask_white_hsv = cv2.inRange(hsv, lower_white, upper_white)
        
        # ماسک زرد (پلاک زرد)
        lower_yellow = np.array([15, 60, 120])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        mask = cv2.bitwise_or(mask_white_hsv, mask_yellow)
        mask = cv2.bitwise_or(mask, bright)
        
        # مورفولوژی: بستن حفره‌ها و حذف نویز
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 800:  # حداقل اندازه معقول برای پلاک
                continue
            
            # مستطیل با حداقل مساحت تا پلاک کج هم تشخیص داده شود
            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect
            if rw <= 0 or rh <= 0:
                continue
            
            # نسبت ابعاد پلاک ایرانی تقریباً ۳:۱ تا ۴:۱ (عرض به ارتفاع)
            if rw >= rh:
                aspect = rw / rh
            else:
                aspect = rh / rw
            
            if aspect < 2.2 or aspect > 5.5:
                continue
            
            # حداقل اندازه پیکسلی (عرض و ارتفاع واقعی بعد از چرخش)
            box_w = max(rw, rh)
            box_h = min(rw, rh)
            if box_w < 60 or box_h < 18:
                continue
            
            # زاویه مجاز: پلاک تقریباً افقی (حداکثر کجی معمول)
            if abs(angle) > 75 and abs(angle) < 105:
                angle = angle - 90
            if abs(angle) > 25:
                continue
            
            # کادر محاط (برای برش تصویر)
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            x_roi, y_roi, w_roi_rect, h_roi_rect = cv2.boundingRect(box_points)
            
            # برگرداندن مختصات به فضای کل تصویر خودرو
            x1 = max(0, x_roi)
            y1 = max(0, search_y_start + y_roi)
            x2 = min(w_img, x_roi + w_roi_rect)
            y2 = min(h_img, search_y_start + y_roi + h_roi_rect)
            
            # حاشیه کم برای اینکه لبه پلاک و حروف بریده نشوند
            pad = max(2, int(min(w_roi_rect, h_roi_rect) * 0.08))
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w_img, x2 + pad)
            y2 = min(h_img, y2 + pad)
            
            # امتیاز: ترجیح ناحیهٔ پایین‌تر (یک‌سوم پایین)
            center_y = (y1 + y2) / 2
            lower_third_bonus = 1.0 if center_y >= h_img * (2/3) else (0.7 if center_y >= h_img * 0.5 else 0.4)
            area_score = min(area / 5000, 1.0)
            aspect_ok = 1.0 if 2.8 <= aspect <= 4.5 else 0.8
            score = lower_third_bonus * 0.5 + area_score * 0.3 + aspect_ok * 0.2
            
            plates.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': min(0.5 + score * 0.4, 0.95),
                'area': area,
                '_score': score,
                'angle': angle,
            })
        
        # مرتب‌سازی بر اساس امتیاز (ترجیح یک‌سوم پایین و اندازه مناسب)
        plates.sort(key=lambda x: (x['_score'], x['area']), reverse=True)
        for p in plates:
            p.pop('_score', None)
        
        return plates
    
    def read_plate_text(self, plate_image):
        """
        خواندن متن پلاک با استفاده از OCR
        
        Args:
            plate_image: تصویر ناحیه پلاک
        
        Returns:
            str: متن پلاک یا None
        """
        reader = self._get_ocr_reader()
        if reader is None:
            return None
        
        # پیش‌پردازش تصویر برای بهبود OCR
        processed = self._preprocess_plate_image(plate_image)
        
        try:
            # خواندن متن
            results = reader.readtext(processed)
            
            if not results:
                return None
            
            # استخراج متن از نتایج
            texts = [text for (bbox, text, confidence) in results if confidence > 0.3]
            full_text = ' '.join(texts)
            
            return full_text
            
        except Exception as e:
            print(f"[!] خطا در OCR: {e}")
            return None
    
    def _preprocess_plate_image(self, image):
        """
        پیش‌پردازش تصویر پلاک برای بهبود OCR (شامل اصلاح کجی احتمالی).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # اصلاح کجی: تخمین زاویه از خطوط افقی متن پلاک
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=min(w, h) // 4, minLineLength=w // 4, maxLineGap=10)
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) > 5:
                    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if abs(ang) < 15:
                        angles.append(ang)
        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:
                M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
                gray = cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # بزرگ‌نمایی برای OCR بهتر
        scale_factor = 2
        gray = cv2.resize(gray, (w * scale_factor, h * scale_factor))
        
        # threshold و کاهش نویز
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        return denoised
    
    def parse_iranian_plate(self, text):
        """
        تجزیه و تحلیل متن پلاک ایرانی
        فرمت: 2 رقم + حرف فارسی + 3 رقم + 2 رقم
        
        Args:
            text: متن خوانده شده از پلاک
        
        Returns:
            dict: اجزای پلاک یا None
        """
        if not text:
            return None
        
        # حذف فاصله‌های اضافی
        text = text.strip()
        
        # الگوهای مختلف برای پلاک ایرانی
        # الگو 1: 12 الف 345 67
        pattern1 = r'(\d{2})\s*([آ-ی])\s*(\d{3})\s*(\d{2})'
        
        # جستجو با regex
        match = re.search(pattern1, text)
        
        if match:
            return {
                'full_text': text,
                'part1': match.group(1),  # 2 رقم اول
                'letter': match.group(2),  # حرف
                'part2': match.group(3),  # 3 رقم وسط
                'part3': match.group(4),  # 2 رقم آخر
                'formatted': f"{match.group(1)} {match.group(2)} {match.group(3)} - {match.group(4)}"
            }
        
        # اگر الگو مطابقت نداشت، سعی می‌کنیم اجزا را جدا کنیم
        return self._parse_flexible(text)
    
    def _parse_flexible(self, text):
        """
        تجزیه انعطاف‌پذیر متن پلاک
        
        Args:
            text: متن پلاک
        
        Returns:
            dict یا None
        """
        # استخراج ارقام و حروف
        digits = re.findall(r'\d+', text)
        letters = re.findall(r'[آ-ی]', text)
        
        # بررسی تعداد ارقام و حروف
        all_digits = ''.join(digits)
        
        if len(all_digits) >= 7 and len(letters) >= 1:
            # تلاش برای تطبیق با فرمت
            try:
                part1 = all_digits[0:2]
                part2 = all_digits[2:5]
                part3 = all_digits[5:7]
                letter = letters[0]
                
                return {
                    'full_text': text,
                    'part1': part1,
                    'letter': letter,
                    'part2': part2,
                    'part3': part3,
                    'formatted': f"{part1} {letter} {part2} - {part3}"
                }
            except:
                pass
        
        return None
    
    def process_vehicle(self, image, vehicle_bbox):
        """
        پردازش کامل یک خودرو: تشخیص و خواندن پلاک
        
        Args:
            image: تصویر کامل
            vehicle_bbox: مختصات خودرو (x1, y1, x2, y2)
        
        Returns:
            dict: نتایج پردازش پلاک
        """
        x1, y1, x2, y2 = vehicle_bbox
        
        # برش تصویر خودرو
        vehicle_image = image[y1:y2, x1:x2]
        
        if vehicle_image.size == 0:
            return None
        
        # تشخیص ناحیه پلاک
        plates = self.detect_plate_region(vehicle_image)
        
        if not plates:
            return {
                'has_plate': False,
                'plate_count': 0,
                'plates': []
            }
        
        # پردازش هر پلاک
        results = []
        for plate in plates[:2]:  # حداکثر 2 پلاک
            px1, py1, px2, py2 = plate['bbox']
            plate_image = vehicle_image[py1:py2, px1:px2]
            
            # خواندن متن
            text = self.read_plate_text(plate_image)
            
            # تجزیه متن
            parsed = self.parse_iranian_plate(text) if text else None
            
            # محاسبه مختصات در تصویر اصلی
            absolute_bbox = (
                x1 + px1,
                y1 + py1,
                x1 + px2,
                y1 + py2
            )
            
            results.append({
                'bbox': absolute_bbox,
                'relative_bbox': (px1, py1, px2, py2),
                'confidence': plate['confidence'],
                'text': text,
                'parsed': parsed
            })
        
        return {
            'has_plate': len(results) > 0,
            'plate_count': len(results),
            'plates': results
        }


def main():
    """تابع تست"""
    import sys
    
    if len(sys.argv) < 2:
        print("استفاده: python plate_reader.py <image_path>")
        return
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"خطا: نمی‌توان تصویر '{image_path}' را خواند.")
        return
    
    reader = IranianPlateReader()
    
    # فرض می‌کنیم کل تصویر یک خودرو است
    h, w = image.shape[:2]
    result = reader.process_vehicle(image, (0, 0, w, h))
    
    print("\nنتیجه:")
    print(f"پلاک شناسایی شد: {result['has_plate']}")
    print(f"تعداد پلاک: {result['plate_count']}")
    
    for i, plate in enumerate(result['plates'], 1):
        print(f"\nپلاک {i}:")
        print(f"  متن: {plate['text']}")
        if plate['parsed']:
            print(f"  فرمت شده: {plate['parsed']['formatted']}")


if __name__ == "__main__":
    main()
