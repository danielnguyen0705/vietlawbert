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
# 1. CẤU HÌNH MASTER LOGGER & ĐƯỜNG DẪN ĐỘNG
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

JSON_DIR = os.path.join(BASE_DIR, "json")
os.makedirs(JSON_DIR, exist_ok=True)

TODAY_STR = datetime.now().strftime("%Y-%m-%d")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs", TODAY_STR)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, "log_master.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [%(levelname)s] | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding='utf-8', mode="a"),
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
        if "app" in command or "generator" in command:
            # Streamlit/App chạy tương tác nên không capture output
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
    
    if not os.path.exists(failed_file_path):
        logger.info("[KIEM TRA] Khong co link loi nao can giai cuu.")
        return True

    while os.path.exists(failed_file_path) and round_count <= MAX_RETRIES:
        with open(failed_file_path, 'r', encoding='utf-8') as f:
            lines = [line for line in f if line.strip()]
            error_count = len(lines)
            
        if error_count == 0:
            break
            
        logger.info(f"==> PHAT HIEN {error_count} LINK LOI. Kich hoat Rescue Vong {round_count}/{MAX_RETRIES}...")
        run_command("scrapy crawl rescue_spider", f"CHAY RESCUE SPIDER (VONG {round_count})")
        round_count += 1
        
    logger.info("[BAO CAO] Da hoan tat cac vong giai cuu!")
    return True

# ==========================================
# 3. MENU DIEU PHOI
# ==========================================
def main():
    logger.info("--- KHOI DONG HE THONG DIEU PHOI VIETLAWBERT ---")
    
    while True:
        print("\n" + "="*65)
        print("      VIETLAWBERT - TRINH DIEU PHOI DU LIEU THONG MINH     ")
        print("="*65)
        print(" 0. Bat dau He thong Docker (Milvus, Neo4j, MongoDB)")
        print(" 1. Cao du lieu (Metadata + Luoc do + HTML)")
        print(" 2. Giai cuu link loi (Rescue Spider)")
        print(" 3. Chuyen doi HTML sang Markdown")
        print(" 4. Bam Chunk & Sinh Ngu Canh (Gemini Contextualizer)")
        print(" 5. Nap du lieu vao DB (Milvus & Neo4j)")
        print(" 6. CHAY TOAN BO QUY TRINH (Auto tu A-Z, XONG MO CHAT LUON)")
        print(" 7. Khoi dong Chatbot (Giao dien App)")
        print(" q. Thoat")
        
        choice = input("\nDaniel chon buoc nao? ").strip().lower()

        if choice == '0':
            run_command("docker-compose up -d", "KHOI DONG DOCKER CONTAINERS")
            logger.info("Vui long doi 10-15s de cac Database san sang...")

        elif choice == '1':
            run_command("scrapy crawl law_spider", "GIAI DOAN 1: Cao du lieu")

        elif choice == '2':
            run_rescue_auto()

        elif choice == '3':
            run_command("python -X utf8 -m preprocess.html_to_md", "GIAI DOAN 2: Chuyen doi Markdown")

        elif choice == '4':
            run_command("python -X utf8 -m preprocess.contextualizer", "GIAI DOAN 3: Bam Chunk & Contextualize")

        elif choice == '5':
            run_command("python -X utf8 -m database.milvus_client", "NAP MILVUS")
            run_command("python -X utf8 -m database.neo4j_client", "NAP NEO4J")
        
        elif choice == '6':
            logger.info("\n" + "*"*50)
            logger.info("[AUTO] BAT DAU PIPELINE TU DONG HOAN TOAN")
            logger.info("*"*50 + "\n")
            
            # Buoc 0: Dam bao Docker da chay
            run_command("docker-compose up -d", "BUOC 0: Dam bao Docker Infrastructure dang chay")
            
            # Buoc 1 & 2: Crawl & Rescue
            run_command("scrapy crawl law_spider", "BUOC 1: Cao du lieu ban dau")
            run_rescue_auto()
            
            # Buoc 3: Chuyen doi Markdown
            if run_command("python -X utf8 -m preprocess.html_to_md", "BUOC 3: Chuyen Markdown"):
                # Buoc 4: Contextualize
                if run_command("python -X utf8 -m preprocess.contextualizer", "BUOC 4: Bam Chunk & Contextualize"):
                    # Buoc 5: Ingestion
                    run_command("python -X utf8 -m database.milvus_client", "BUOC 5A: Nap vao Milvus")
                    run_command("python -X utf8 -m database.neo4j_client", "BUOC 5B: Nap vao Neo4j")
                    
                    logger.info("\n🎉 [THANH CONG] DU LIEU DA SAN SANG! DANG MO CHATBOT...\n")
                    run_command("python -X utf8 app.py", "BUOC 6: KHOI DONG GIAO DIEN CHATBOT")
                else:
                    logger.error("⛔ [DUNG] Loi tai buoc Contextualize.")
            else:
                logger.error("⛔ [DUNG] Loi tai buoc Markdown.")

        elif choice == '7':
            run_command("python -X utf8 app.py", "KHOI DONG CHATBOT")

        elif choice == 'q':
            logger.info("Tam biet Daniel!")
            break

if __name__ == "__main__":
    main()