import subprocess
import os
import sys
import logging
from datetime import datetime

# ============================================================
# CAU HINH UTF-8 CHO TOAN HE THONG
# ============================================================
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ==========================================
# 1. CAU HINH MASTER LOGGER
# ==========================================
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/logs'))
os.makedirs(LOG_DIR, exist_ok=True)

today_str = datetime.now().strftime("%Y-%m-%d")
MASTER_LOG_FILE = os.path.join(LOG_DIR, f'master_{today_str}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [%(levelname)s] | %(message)s',
    handlers=[
        logging.FileHandler(MASTER_LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VietLawBERT_Master")

# ==========================================
# 2. HAM CHAY LENH (DA LOAI BO ICON)
# ==========================================
def run_command(command, description):
    logger.info("-" * 60)
    logger.info(f"[BAT DAU] {description}")
    logger.info(f"[LENH] {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            text=True, 
            capture_output=True,
            encoding='utf-8',
            errors='replace' 
        )
        
        if result.returncode == 0:
            logger.info(f"[HOAN THANH] {description}")
            if result.stdout:
                summary = result.stdout[-500:].strip()
                logger.info(f"[LOG] Ket qua cuoi:\n{summary}")
            return True
        else:
            logger.error(f"[LOI] {description} (Ma loi: {result.returncode})")
            if result.stderr:
                logger.error(f"[CHITIET] {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"[SU CO] Loi he thong khi chay: {description}\nChi tiet: {str(e)}")
        return False

# ==========================================
# HAM CHAY RESCUE TU DONG
# ==========================================
def run_rescue_auto():
    # Sử dụng đường dẫn tuyệt đối giống hệt trong file spider
    failed_file_path = r"D:\Daniel_Nguyen\nckh_project\law_dataset\json\failed_links.jsonl"
    
    # Số lần lặp tối đa để tránh bị kẹt nếu server sập hẳn
    MAX_RETRIES = 3 
    round_count = 1
    
    logger.info("[KIEM TRA] Dang check file failed_links.jsonl...")
    
    while os.path.exists(failed_file_path) and round_count <= MAX_RETRIES:
        # Đếm số lượng link lỗi hiện tại
        with open(failed_file_path, 'r', encoding='utf-8') as f:
            error_count = sum(1 for line in f if line.strip())
            
        if error_count == 0:
            break
            
        logger.info(f"==> PHAT HIEN {error_count} LINK LOI. Kich hoat Rescue Vong {round_count}/{MAX_RETRIES}...")
        
        # Chạy nhện cứu hộ
        run_command("scrapy crawl rescue_spider", f"CHAY RESCUE SPIDER (VONG {round_count})")
        
        round_count += 1
        
    # Kiểm tra lại lần cuối sau khi hết vòng lặp
    if os.path.exists(failed_file_path):
        with open(failed_file_path, 'r', encoding='utf-8') as f:
            remaining_errors = sum(1 for line in f if line.strip())
        if remaining_errors > 0:
            logger.warning(f"[BAO CAO] Vẫn còn {remaining_errors} link chưa thể cứu sau {MAX_RETRIES} vòng. Server có thể đang chặn IP hoặc lỗi nặng.")
            return False
    
    logger.info("[BAO CAO] Đã dọn sạch 100% link lỗi!")
    return True


# ==========================================
# 3. MENU DIEU PHOI
# ==========================================
def main():
    logger.info("--- KHOI DONG HE THONG DIEU PHOI VIETLAWBERT ---")
    
    while True:
        print("\n" + "="*60)
        print("   VIETLAWBERT - TRINH DIEU PHOI DU LIEU THONG MINH   ")
        print("="*60)
        print(" 1. Cao du lieu (Metadata + Luoc do + HTML)")
        print(" 2. Giai cuu link loi (Chay rieng Rescue Spider)")
        print(" 3. Chuyen doi HTML sang Markdown")
        print(" 4. Kiem tra toan ven du lieu (Audit)")
        print(" 5. Nap du lieu vao DB (Milvus/Neo4j)")
        print(" 6. CHAY TOAN BO QUY TRINH (Cao -> Giai Cuu -> Markdown)")
        print(" q. Thoat")
        
        choice = input("\nDaniel chon buoc nao? ").strip().lower()

        if choice == '1':
            run_command("scrapy crawl law_spider", "GIAI DOAN 1: Cao du lieu")
            
        elif choice == '2':
            run_rescue_auto()
            
        elif choice == '3':
            run_command("python -X utf8 -m preprocess.html_to_md", "GIAI DOAN 2: Chuyen doi Markdown")

        elif choice == '4':
            logger.info("[KIEM TOAN] Dang quet thu muc data/raw/...")
            try:
                html_path = os.path.join(LOG_DIR, '../raw/html')
                if os.path.exists(html_path):
                    html_count = len([f for f in os.listdir(html_path) if f.endswith('.html')])
                    logger.info(f"[THONG KE] Dang co {html_count} file HTML.")
                else:
                    logger.warning("[CANH BAO] Chua co du lieu HTML.")
            except Exception as e:
                logger.error(f"[LOI] Khong the kiem toan: {e}")

        elif choice == '5':
            print("[THONG BAO] Chuc nang dang phat trien.")

        elif choice == '6':
            logger.info("[AUTO] BAT DAU PIPELINE TU DONG HOAN TOAN")
            
            # Bước 1: Cào lần đầu
            run_command("scrapy crawl law_spider", "BUOC 1: Cao du lieu ban dau")
            
            # Bước 2: Tự động phát hiện và chạy Rescue nhiều vòng nếu cần
            run_rescue_auto()
            
            # Bước 3: Đã sạch dữ liệu thì chuyển sang Markdown
            if run_command("python -X utf8 -m preprocess.html_to_md", "BUOC 3: Chuyen Markdown"):
                logger.info("[THANH CONG] PIPELINE HOAN TAT TAT CA CAC BUOC.")

        elif choice == 'q':
            logger.info("Tat he thong. Tam biet Daniel!")
            break
        else:
            print("Lua chon khong hop le.")

if __name__ == "__main__":
    main()