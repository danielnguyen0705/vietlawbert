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
# 1. CAU HINH MASTER LOGGER & ĐƯỜNG DẪN ĐỘNG
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(BASE_DIR, 'data', 'logs')
JSON_DIR = os.path.join(BASE_DIR, 'json')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

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
# 2. HAM CHAY LENH
# ==========================================
def run_command(command, description):
    logger.info("-" * 60)
    logger.info(f"[BAT DAU] {description}")
    logger.info(f"[LENH] {command}")
    
    try:
        # Riêng cái chatbot (app) thì không capture output để nó tương tác được trên Terminal
        if "app" in command or "generator" in command:
            result = subprocess.run(command, shell=True)
            return result.returncode == 0
            
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
    failed_file_path = os.path.join(JSON_DIR, "failed_links.jsonl")
    MAX_RETRIES = 3 
    round_count = 1
    
    logger.info("[KIEM TRA] Dang check file failed_links.jsonl...")
    
    while os.path.exists(failed_file_path) and round_count <= MAX_RETRIES:
        with open(failed_file_path, 'r', encoding='utf-8') as f:
            error_count = sum(1 for line in f if line.strip())
            
        if error_count == 0:
            break
            
        logger.info(f"==> PHAT HIEN {error_count} LINK LOI. Kich hoat Rescue Vong {round_count}/{MAX_RETRIES}...")
        run_command("scrapy crawl rescue_spider", f"CHAY RESCUE SPIDER (VONG {round_count})")
        round_count += 1
        
    if os.path.exists(failed_file_path):
        with open(failed_file_path, 'r', encoding='utf-8') as f:
            remaining_errors = sum(1 for line in f if line.strip())
        if remaining_errors > 0:
            logger.warning(f"[BAO CAO] Van con {remaining_errors} link chua the cuu sau {MAX_RETRIES} vong.")
            return False 
    
    logger.info("[BAO CAO] Da don sach 100% link loi!")
    return True

# ==========================================
# 3. MENU DIEU PHOI
# ==========================================
def main():
    logger.info("--- KHOI DONG HE THONG DIEU PHOI VIETLAWBERT ---")
    
    while True:
        print("\n" + "="*65)
        print("     VIETLAWBERT - TRINH DIEU PHOI DU LIEU THONG MINH     ")
        print("="*65)
        print(" 1. Cao du lieu (Metadata + Luoc do + HTML)")
        print(" 2. Giai cuu link loi (Chay rieng Rescue Spider)")
        print(" 3. Chuyen doi HTML sang Markdown")
        print(" 4. Bam Chunk & Sinh Ngu Canh (Contextualizer bang Gemini)")
        print(" 5. Nap du lieu vao DB (Milvus & Neo4j)")
        print(" 6. CHAY TOAN BO QUY TRINH (Auto tu A-Z, XONG MO CHAT LUON)")
        print(" 7. Khoi dong Chatbot (Chi bat giao dien Chat)")
        print(" q. Thoat")
        
        choice = input("\nDaniel chon buoc nao? ").strip().lower()

        if choice == '1':
            run_command("scrapy crawl law_spider", "GIAI DOAN 1: Cao du lieu")
        elif choice == '2':
            run_rescue_auto()
        elif choice == '3':
            run_command("python -X utf8 -m preprocess.html_to_md", "GIAI DOAN 2: Chuyen doi Markdown")
        elif choice == '4':
            run_command("python -X utf8 -m preprocess.contextualizer", "GIAI DOAN 3: Bam Chunk & Contextualize")
        elif choice == '5':
            logger.info("[NAP DU LIEU] Dang nhoi du lieu vao he thong Databases...")
            success_milvus = run_command("python -X utf8 -m database.milvus_client", "NAP MILVUS (Vector DB)")
            success_neo4j = run_command("python -X utf8 -m database.neo4j_client", "NAP NEO4J (Graph DB)")
            if success_milvus and success_neo4j:
                logger.info("✅ NAP DU LIEU THANH CONG!")
        
        elif choice == '6':
            logger.info("\n" + "*"*50)
            logger.info("[AUTO] BAT DAU PIPELINE TU DONG HOAN TOAN")
            logger.info("Ong cu di ngu di, de may moc lo! 🚀")
            logger.info("*"*50 + "\n")
            
            run_command("scrapy crawl law_spider", "BUOC 1: Cao du lieu ban dau")
            run_rescue_auto()
            
            if run_command("python -X utf8 -m preprocess.html_to_md", "BUOC 3: Chuyen Markdown"):
                if run_command("python -X utf8 -m preprocess.contextualizer", "BUOC 4: Bam Chunk & Contextualize"):
                    run_command("python -X utf8 -m database.milvus_client", "BUOC 5A: Nap vao Milvus")
                    run_command("python -X utf8 -m database.neo4j_client", "BUOC 5B: Nap vao Neo4j")
                    
                    logger.info("\n🎉 [THANH CONG] DU LIEU DA SAN SANG! DANG MO CHATBOT...\n")
                    
                    # BUOC CUOI: TU DONG MO CHATBOT NAY!
                    run_command("python -X utf8 app.py", "BUOC 6: KHOI DONG GIAO DIEN CHATBOT")
                else:
                    logger.error("⛔ [DUNG PIPELINE] Buoc 4 Contextualize bi loi.")
            else:
                logger.error("⛔ [DUNG PIPELINE] Buoc 3 Markdown bi loi.")

        elif choice == '7':
            logger.info("🚀 KHOI DONG CHATBOT...")
            # Sửa 'app.py' thành tên file UI chính xác của ông nếu cần (ví dụ: src/rag/app.py)
            run_command("python -X utf8 app.py", "KHOI DONG GIAO DIEN CHATBOT")

        elif choice == 'q':
            logger.info("Tat he thong. Tam biet Daniel!")
            break
        else:
            print("Lua chon khong hop le.")

if __name__ == "__main__":
    main()