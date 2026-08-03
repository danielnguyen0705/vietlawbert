import subprocess
import os
import sys
import logging
from datetime import datetime

# ============================================================
# CAU HINH UTF-8 & PYTHONPATH CHO TOAN HE THONG
# ============================================================
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = current_dir + os.pathsep + os.environ.get("PYTHONPATH", "")

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

JSON_DIR = os.path.join(BASE_DIR, "data", "json")
os.makedirs(JSON_DIR, exist_ok=True)

TODAY_STR = datetime.now().strftime("%Y-%m-%d")
LOG_DIR = os.path.join(BASE_DIR, "logs", TODAY_STR)
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

PYTHON_EXEC = sys.executable
VENV_BIN = os.path.dirname(PYTHON_EXEC)
SCRAPY_EXEC = os.path.join(VENV_BIN, "scrapy")

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def run_command(command, description, cwd=None):
    logger.info("-" * 60)
    logger.info(f"[BAT DAU] {description}")

    if command.startswith("python"):
        command = command.replace("python", f'"{PYTHON_EXEC}"', 1)

    logger.info(f"[LENH] {command}  (cwd={cwd or 'inherit'})")

    try:
        if "app.py" in command or "chainlit" in command:
            result = subprocess.run(command, shell=True, cwd=cwd)
            return result.returncode == 0

        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
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
        run_command(f'"{SCRAPY_EXEC}" crawl rescue_spider', f"CHAY RESCUE SPIDER (VONG {round_count})", cwd=CURRENT_DIR)
        round_count += 1

    logger.info("[BAO CAO] Da hoan tat cac vong giai cuu!")
    return True


def main():
    logger.info("--- KHOI DONG HE THONG DIEU PHOI VIETLAWBERT ---")

    while True:
        print("\n" + "=" * 65)
        print("      VIETLAWBERT - TRINH DIEU PHOI DU LIEU THONG MINH     ")
        print("=" * 65)
        print(" 0. Bat dau He thong Docker (Milvus, Neo4j, MongoDB, Redpanda)")
        print(" 1. Cao du lieu (Metadata + Luoc do + HTML)")
        print(" 2. Giai cuu link loi (Rescue Spider)")
        print(" 3. Chuyen doi HTML sang Markdown")
        print(" 4. Chay Kafka Consumer (Doc tu Kafka, Nap Milvus + Neo4j)")
        print(" 5. Sinh Triplet Dataset (GG-SLM Semi-Hard Negative)")
        print(" 6. Fine-tune Embedding (Matryoshka Learning)")
        print(" 7. CHAY TOAN BO QUY TRINH (Auto tu A-Z)")
        print(" 8. Khoi dong Chatbot (Giao dien Chainlit)")
        print(" q. Thoat")

        choice = input("\nDaniel chon buoc nao? ").strip().lower()

        if choice == '0':
            run_command("docker compose up -d", "KHOI DONG DOCKER CONTAINERS", cwd=PROJECT_ROOT)
            logger.info("Vui long doi 10-15s de cac Database san sang...")

        elif choice == '1':
            run_command(f'"{SCRAPY_EXEC}" crawl law_spider', "GIAI DOAN 1: Cao du lieu", cwd=CURRENT_DIR)

        elif choice == '2':
            run_rescue_auto()

        elif choice == '3':
            run_command("python -X utf8 -m preprocess.html_to_md", "GIAI DOAN 2: Chuyen doi Markdown")

        elif choice == '4':
            logger.info("[INFO] Consumer doc tu Kafka, nhan message tu Crawler/Contextualizer")
            logger.info("[INFO] Bam chunk + Sinh ngu canh (Ollama) + Bulk Insert Milvus & Neo4j")
            run_command("python -X utf8 -m streaming.consumer", "CHAY KAFKA CONSUMER (PIPELINE LIEN TUUC)")

        elif choice == '5':
            run_command("python -X utf8 -m training.generate_training_data --tau 0.1", "SINH TRIPLET DATASET (GG-SLM)")

        elif choice == '6':
            run_command("python -X utf8 -m training.fine_tune", "FINE-TUNE EMBEDDING MODEL (MATRYOSHKA)")

        elif choice == '7':
            logger.info("\n" + "*" * 50)
            logger.info("[AUTO] BAT DAU PIPELINE TU DONG HOAN TOAN")
            logger.info("*" * 50 + "\n")

            run_command("docker compose up -d", "BUOC 0: Dam bao Docker dang chay", cwd=PROJECT_ROOT)

            run_command(f'"{SCRAPY_EXEC}" crawl law_spider', "BUOC 1: Cao du lieu ban dau", cwd=CURRENT_DIR)
            run_rescue_auto()

            if run_command("python -X utf8 -m preprocess.html_to_md", "BUOC 2: Chuyen Markdown"):
                if run_command("python -X utf8 -m streaming.consumer", "BUOC 3: Chay Consumer (Kafka -> Milvus/Neo4j)"):
                    run_command("python -X utf8 -m training.generate_training_data --tau 0.1", "BUOC 4: Sinh Triplet Dataset")

                    logger.info("\nDU LIEU DA SAN SANG! DANG MO CHATBOT...\n")
                    run_command(f'"{PYTHON_EXEC}" -m chainlit run {os.path.join(BASE_DIR, "chainlit_app.py")} --headless', "BUOC 5: KHOI DONG CHATBOT (CHAINLIT)")
                else:
                    logger.error("[DUNG] Loi tai buoc Consumer.")
            else:
                logger.error("[DUNG] Loi tai buoc Markdown.")

        elif choice == '8':
            run_command(f'"{PYTHON_EXEC}" -m chainlit run {os.path.join(BASE_DIR, "chainlit_app.py")} --headless', "KHOI DONG CHATBOT (CHAINLIT)")

        elif choice == 'q':
            logger.info("Tam biet Daniel!")
            break


if __name__ == "__main__":
    main()
