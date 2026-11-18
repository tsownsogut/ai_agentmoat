from dotenv import load_dotenv
import os

# Tự động tìm và tải file .env từ thư mục hiện tại hoặc các thư mục cha
load_dotenv()

# Bây giờ bạn có thể truy cập biến như bình thường
api_key = os.getenv("GOOGLE_API_KEY")
cse_id = os.getenv("GOOGLE_CSE_ID")