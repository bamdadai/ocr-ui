import asyncio
import aiohttp
import aiofiles
import time
import json
import uuid
import glob  # جدید: برای پیدا کردن فایل‌ها
import os    # جدید: برای کار با مسیر فایل

# --- تنظیمات تست ---
API_URL = "http://localhost:8000/v3/ocr"
WEBHOOK_URL = "http://127.0.0.1:8080/webhook"

# لیستی برای نگهداری نتایج اولیه
results = []

async def send_request(session, file_path, request_num):
    """یک درخواست تکی به اندپوینت OCR ارسال می‌کند."""
    try:
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # تعیین content_type بر اساس پسوند فایل
        if file_ext == '.pdf':
            content_type = '.pdf'
        elif file_ext in ['.jpg', '.jpeg']:
            content_type = '.jpeg'
        elif file_ext == '.png':
            content_type = '.png'
        else:
            print(f"⚠️ فایل {file_name} پسوند نامعتبر دارد و نادیده گرفته شد.")
            return

        metadata = [{
            "guid": f"stress-test-{uuid.uuid4()}",
            "file_type": content_type
        }]

        form_data = aiohttp.FormData()
        async with aiofiles.open(file_path, 'rb') as f:
            file_content = await f.read()
            form_data.add_field(
                'files',
                file_content,
                filename=file_name,
                content_type=content_type
            )
        
        form_data.add_field('metadata', json.dumps(metadata))
        form_data.add_field('webhook_url', WEBHOOK_URL)
        
        async with session.post(API_URL, data=form_data) as response:
            response_json = await response.json()
            if response.status == 200:
                tasks = response_json if isinstance(response_json, list) else response_json.get("tasks", [])
                queued = any(task.get("status") == "queued" for task in tasks) if isinstance(tasks, list) else False
                if queued:
                    print(f"✅ درخواست {request_num} ({file_name}): تسک با موفقیت در صف قرار گرفت.")
                    results.append({"status": "queued", "file": file_name})
                else:
                    print(f"⚠️ درخواست {request_num} ({file_name}): پاسخ بدون تسک معتبر دریافت شد.")
                    results.append({"status": "unexpected", "file": file_name})
            else:
                print(f"❌ درخواست {request_num} ({file_name}): خطا در ارسال. Status: {response.status}")
                results.append({"status": "failed", "file": file_name})

    except Exception as e:
        print(f"🔥 درخواست {request_num}: خطای بحرانی: {e}")
        results.append({"status": "error", "file": file_path})

async def main():
    """تابع اصلی برای پیدا کردن فایل‌ها و ارسال همزمان."""
    
    # جدید: پیدا کردن تمام فایل‌های مورد نظر در پوشه فعلی
    print("در حال جستجو برای فایل‌های .jpg, .png, .pdf ...")
    file_types = ('*.jpg', '*.jpeg', '*.png', '*.pdf')
    file_list = []
    for f_type in file_types:
        file_list.extend(glob.glob(f_type))

    if not file_list:
        print("هیچ فایل عکسی یا PDF در این پوشه پیدا نشد!")
        return
        
    NUM_REQUESTS = len(file_list)
    print(f"تعداد {NUM_REQUESTS} فایل پیدا شد. شروع به ارسال ...\n")
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        # تغییر: به جای یک حلقه ثابت، روی لیست فایل‌های پیدا شده حلقه می‌زنیم
        tasks = [send_request(session, file_path, i + 1) for i, file_path in enumerate(file_list)]
        await asyncio.gather(*tasks)

    end_time = time.time()
    print("\n" + "="*50)
    print(f"تمام {NUM_REQUESTS} درخواست در مدت {end_time - start_time:.2f} ثانیه به صف اضافه شدند.")
    print(f"تعداد تسک‌های موفق در صف: {len([r for r in results if r['status'] == 'queued'])}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
