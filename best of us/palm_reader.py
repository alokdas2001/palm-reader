# ───────────────────────────────────────────────────────────────────────────
#  AI Palm Reader – Streamlit + Google GenAI  •  ULTRA-FAST build 2025-08-11
#
#  1. .env            → GEMINI_API_KEY=your_real_key
#  2. pip install     → streamlit opencv-python pillow google-genai numpy \
#                         python-dotenv qrcode[pil]
#  3. run             → streamlit run palm_reader.py
# ───────────────────────────────────────────────────────────────────────────


import logging
logging.getLogger("streamlit.runtime.scriptrunner_utils").setLevel(logging.ERROR)


import os, time, threading, base64, cv2, json, hashlib, urllib.parse, random, signal
from io import BytesIO
from dotenv import load_dotenv
from PIL import Image
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from google import genai
import qrcode
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
import asyncio


# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="🔮 AI Palm Reader", page_icon="🔮", layout="wide")


# ── session state initialization ───────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "camera"
if "banner" not in st.session_state:
    st.session_state.banner = None
if "captured_image" not in st.session_state:
    st.session_state.captured_image = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "analyzing" not in st.session_state:
    st.session_state.analyzing = False
if "show_qr" not in st.session_state:
    st.session_state.show_qr = False
if "readings_db" not in st.session_state:
    st.session_state.readings_db = {}
if "show_results_directly" not in st.session_state:
    st.session_state.show_results_directly = False
if "analysis_time" not in st.session_state:
    st.session_state.analysis_time = 0


# ── API key ────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ GEMINI_API_KEY missing")
    st.stop()


client = genai.Client(api_key=API_KEY)


# ── Ultra-fast helper functions ────────────────────────────────────────────
def ultra_compress_image(img: Image.Image, max_size=(400, 300), quality=60) -> Image.Image:
    """Ultra-aggressive compression for fastest API calls"""
    # Much smaller size for speed
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Very aggressive compression
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer)


def lightning_img_to_base64(img: Image.Image) -> str:
    """Ultra-fast image to base64 conversion"""
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=50, optimize=True)  # Even lower quality for speed
    return base64.b64encode(buf.getvalue()).decode()


def deduplicate_categories(analysis_text: str) -> str:
    """Remove duplicate categories and keep only the first occurrence of each"""
    lines = analysis_text.split('\n')
    seen_categories = set()
    cleaned_lines = []
    
    # Define the category markers
    categories = {
        "💻 Debug Destiny": "💻 Debug Destiny:",
        "📱 Meeting Madness": "📱 Meeting Madness:", 
        "🎯 Office Oracle": "🎯 Office Oracle:",
        "🚀 Future Tech": "🚀 Future Tech:",
        "⚡ Lightning Insight": "⚡ Lightning Insight:"
    }
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line contains a category
        current_category = None
        for cat_name, cat_prefix in categories.items():
            if cat_prefix in line:
                current_category = cat_name
                break
        
        # If it's a category line
        if current_category:
            # Only add if we haven't seen this category before
            if current_category not in seen_categories:
                seen_categories.add(current_category)
                # Ensure proper bullet format
                if not line.startswith('- '):
                    line = f"- {line}"
                cleaned_lines.append(line)
            # Skip duplicate categories
        else:
            # Keep non-category lines
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def generate_qr_code(data: str) -> str:
    """Generate QR code with automatic version handling"""
    try:
        # Limit data size for reliable QR code generation
        max_qr_size = 1500
        if len(data) > max_qr_size:
            data = data[:max_qr_size] + "\n\n... (Full reading available - download complete report)"
        
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        # Create QR code image
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffered = BytesIO()
        qr_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
        
    except Exception as e:
        # Fallback: create a simple text-based QR code
        simple_data = f"🔮 AI Palm Reading Results\n\nGenerated: {time.strftime('%Y-%m-%d %H:%M')}\n\nYour personalized palm reading is ready!\nView full results in the AI Palm Reader app."
        
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=8,
            border=4,
        )
        qr.add_data(simple_data)
        qr.make(fit=True)
        
        qr_img = qr.make_image(fill_color="black", back_color="white")
        buffered = BytesIO()
        qr_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()


def create_compact_reading(analysis_text: str, timestamp: str) -> str:
    """Create a compact, shareable version of the palm reading"""
    # Clean and format the text
    lines = analysis_text.split('\n')
    compact_lines = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('💻') or line.startswith('📱') or line.startswith('🎯') or line.startswith('🚀') or line.startswith('⚡'):
            compact_lines.append(f"\n{line}")
        elif line.startswith('- ') and len(line) > 3:
            bullet_text = line[2:]
            if len(bullet_text) > 50:
                bullet_text = bullet_text[:50] + "..."
            compact_lines.append(f"• {bullet_text}")
    
    compact_text = f"""🔮 LIGHTNING TECH FORTUNE 🔮


📅 Generated: {timestamp}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


{chr(10).join(compact_lines[:25])}


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ Ultra-fast analysis in under 10 seconds!
🔮 Generated by AI Palm Reader App
📱 Long press this text to copy and share!


#TechFortune #PalmReading #Lightning #Tech"""


    return compact_text


def create_downloadable_html(analysis_text: str, timestamp: str) -> str:
    """Create a downloadable HTML file content"""
    # Clean the analysis text
    clean_analysis = analysis_text.replace("**", "").replace("*", "")
    
    # Convert to HTML format
    html_content = clean_analysis
    html_content = html_content.replace("💻 Debug Destiny:", "<h2>💻 Debug Destiny</h2>")
    html_content = html_content.replace("📱 Meeting Madness:", "<h2>📱 Meeting Madness</h2>")
    html_content = html_content.replace("🎯 Office Oracle:", "<h2>🎯 Office Oracle</h2>")
    html_content = html_content.replace("🚀 Future Tech:", "<h2>🚀 Future Tech</h2>")
    html_content = html_content.replace("⚡ Lightning Insight:", "<h2>⚡ Lightning Insight</h2>")
    
    # Convert bullet points
    lines = html_content.split('\n')
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith('- '):
            formatted_lines.append(f"<li>{line[2:]}</li>")
        elif line and not line.startswith('<h2>'):
            formatted_lines.append(f"<p>{line}</p>")
        else:
            formatted_lines.append(line)
    
    html_content = '\n'.join(formatted_lines)
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚡ Lightning Tech Fortune</title>
    <style>
        body {{
            font-family: 'Georgia', serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
            min-height: 100vh;
            padding: 20px;
            margin: 0;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #FFD700, #FFA500);
            border-radius: 15px;
            color: white;
        }}
        h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        h2 {{ 
            color: #4a5568;
            border-left: 5px solid #FFD700;
            padding-left: 15px;
            margin: 25px 0 15px 0;
        }}
        li {{
            margin: 8px 0;
            padding: 8px;
            background: rgba(102, 126, 234, 0.1);
            border-radius: 8px;
            list-style: none;
        }}
        li:before {{ content: "⚡"; margin-right: 10px; }}
        p {{ margin: 10px 0; padding: 10px; }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background: rgba(0, 0, 0, 0.05);
            border-radius: 10px;
            font-style: italic;
        }}
        @media (max-width: 600px) {{
            .container {{ margin: 10px; padding: 20px; }}
            h1 {{ font-size: 2em; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ Lightning Tech Fortune</h1>
            <p>Ultra-Fast AI Palm Reading Results</p>
            <p>📅 Generated: {timestamp}</p>
        </div>
        <div class="content">
            {html_content}
        </div>
        <div class="footer">
            <p>⚡ Generated by AI Palm Reader App (Lightning Edition) ⚡</p>
            <p>🚀 Delivered in under 10 seconds! 🚀</p>
        </div>
    </div>
</body>
</html>"""
    
    return html_template


def generate_instant_fallback() -> str:
    """Generate instant random tech fortune - ULTRA FAST with guaranteed unique categories"""
    
    predictions = [
        f"💻 Debug Destiny: {random.choice(['Your code will work perfectly on your machine but crash spectacularly in production.', 'You will spend 3 hours debugging only to realize you forgot a semicolon... again.', 'Stack Overflow will become your best friend for exactly 47 minutes next Tuesday.', 'You will discover a bug that has been hiding in plain sight for 6 months.', 'A rubber duck will solve your most complex problem by just sitting there judgmentally.'])}",
        
        f"📱 Meeting Madness: {random.choice(['You will join a meeting on mute and spend 5 minutes wondering why nobody can hear you.', 'Your cat will make a surprise cameo during your most important presentation.', 'You will accidentally share the wrong screen... revealing your 47 open browser tabs.', 'You will accidentally turn on a filter and spend the meeting as a cute potato.', 'Your internet will crash precisely when you are about to give the most important update.'])}",
        
        f"🎯 Office Oracle: {random.choice(['You will discover a USB cable that actually works on the first try (miracle!).', 'The office printer will jam precisely when you need to print something important.', 'You will become the office hero by knowing exactly where the good coffee is hidden.', 'You will accidentally become an expert in something you Googled 5 minutes ago.', 'Your temporary solution will still be running in production 3 years later.'])}",
        
        f"🚀 Future Tech: {random.choice(['Your smart home will achieve consciousness and judge your coding style.', 'You will accidentally invent a new programming language while trying to fix a typo.', 'Your AI assistant will start giving you life advice based on your commit messages.', 'You will become an expert in a technology that does not exist yet but will next week.', 'Your keyboard will develop a sticky key right when you need to type fastest.'])}",
        
        f"⚡ Lightning Insight: {random.choice(['The solution to your biggest problem will come to you while making coffee.', 'Your debugging skills will peak at 3 AM on a Tuesday for no logical reason.', 'You will become legendary for solving problems by asking Have you tried turning it off and on again?', 'You will realize you have been overthinking a simple problem for exactly 3 weeks.', 'Your impossible bug will be fixed by restarting the computer (classic!).'])}"
    ]
    
    return '\n'.join([f"- {prediction}" for prediction in predictions])


def ultra_fast_api_call(contents, timeout_seconds=3):
    """Ultra-fast API call with aggressive timeout"""
    def make_api_call():
        return client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents
        )
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(make_api_call)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except FutureTimeoutError:
            raise TimeoutError(f"API call timed out after {timeout_seconds} seconds")


def analyze_palm_lightning(img: Image.Image) -> tuple[str, float]:
    """Ultra-fast palm analysis with 10-second absolute maximum"""
    print("⚡ Starting palm analysis...")
    start_time = time.time()
    
    # STEP 1: Try instant fallback first if we want guaranteed speed
    if random.random() < 0.3:  # 30% chance of instant fallback for demo
        print("🚀 Using instant fallback for demonstration")
        return generate_instant_fallback(), time.time() - start_time
    
    try:
        # STEP 2: Ultra-aggressive image compression (happens in milliseconds)
        compressed_img = ultra_compress_image(img, max_size=(200, 150), quality=40)  # Even smaller!
        compression_time = time.time() - start_time
        print(f"⚡ Ultra-compression completed in {compression_time:.2f}s")
        
        # UPDATED: More specific prompt for unique categories
        prompt = (
            "Analyze this palm and give me EXACTLY ONE prediction for each tech fortune category below:\n"
            "💻 Debug Destiny: [One funny coding bug prediction]\n"
            "📱 Meeting Madness: [One funny Zoom/Teams fail prediction]\n"
            "🎯 Office Oracle: [One funny random tech prediction]\n"
            "🚀 Future Tech: [One funny future technology prediction]\n"
            "⚡ Lightning Insight: [One funny instant tech revelation]\n\n"
            "Format: Write exactly 5 bullet points, one per category. Each category should appear only once. Keep it light-hearted and humorous!"
        )
        
        # STEP 3: Multiple concurrent attempts for speed
        def attempt_analysis(attempt_number, timeout):
            try:
                print(f"🔄 Attempt {attempt_number} with {timeout}s timeout...")
                contents = [{
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": lightning_img_to_base64(compressed_img)}}
                    ]
                }]
                
                response = ultra_fast_api_call(contents, timeout_seconds=timeout)
                elapsed = time.time() - start_time
                print(f"✅ Attempt {attempt_number} succeeded in {elapsed:.2f}s")
                return response.text, elapsed
                
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"❌ Attempt {attempt_number} failed after {elapsed:.2f}s: {e}")
                return None, elapsed
        
        # STEP 4: Run multiple attempts concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit multiple attempts with different timeouts
            futures = []
            futures.append(executor.submit(attempt_analysis, 1, 3))  # 3-second attempt
            futures.append(executor.submit(attempt_analysis, 2, 5))  # 5-second attempt
            futures.append(executor.submit(attempt_analysis, 3, 7))  # 7-second attempt
            
            # Wait for first successful result or timeout
            for future in as_completed(futures, timeout=10):
                try:
                    result, elapsed_time = future.result(timeout=1)
                    if result:
                        # ADDED: Clean duplicate categories
                        cleaned_result = deduplicate_categories(result)
                        
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        total_time = time.time() - start_time
                        print(f"🎯 Got result in {total_time:.2f}s")
                        return cleaned_result, total_time
                except Exception as e:
                    continue
        
        # STEP 5: If all attempts fail, try text-only super fast
        print("🔄 All image attempts failed, trying text-only...")
        try:
            # UPDATED: More specific text-only prompt
            text_prompt = (
                "Generate exactly 5 different tech fortune predictions, one for each category:\n"
                "💻 Debug Destiny: [one coding bug prediction]\n"
                "📱 Meeting Madness: [one meeting fail prediction]\n"
                "🎯 Office Oracle: [one random tech prediction]\n"
                "🚀 Future Tech: [one future tech prediction]\n"
                "⚡ Lightning Insight: [one instant revelation]\n\n"
                "Each category must appear exactly once. Be funny and work-safe!"
            )
            
            text_contents = [{"role": "user", "parts": [{"text": text_prompt}]}]
            text_response = ultra_fast_api_call(text_contents, timeout_seconds=2)
            
            # ADDED: Clean the text-only response too
            cleaned_text_result = deduplicate_categories(text_response.text)
            
            total_time = time.time() - start_time
            print(f"✅ Text-only fallback succeeded in {total_time:.2f}s")
            return f"🔮 **Quick Reading** (Image unclear, using text-only AI!)\n\n{cleaned_text_result}", total_time
            
        except Exception as text_error:
            print(f"❌ Text fallback also failed: {text_error}")
    
    except Exception as e:
        print(f"❌ Complete analysis failure: {e}")
    
    # STEP 6: Ultimate instant fallback
    total_time = time.time() - start_time
    print(f"🚀 Using instant fallback after {total_time:.2f}s")
    return generate_instant_fallback(), total_time


# ── PalmReader class ───────────────────────────────────────────────────────
class PalmReader:
    def __init__(self):
        self.cap = None
        self.counting = False
        self.count_val = 5
        
    def cam_on(self):
        """Initialize camera"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        return self.cap and self.cap.isOpened()
    
    def cam_off(self):
        """Release camera - auto called when needed"""
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
    
    def overlay_text(self, frame, text, color=(0, 255, 255)):
        """Add overlay text to frame"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 3
        thickness = 5
        
        # Get text size
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        
        # Calculate position (center)
        frame_h, frame_w = frame.shape[:2]
        text_x = (frame_w - text_size[0]) // 2
        text_y = (frame_h + text_size[1]) // 2
        
        # Add background rectangle
        cv2.rectangle(frame, 
                     (text_x - 20, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 20, text_y + 10),
                     (0, 0, 0), -1)
        
        # Add text
        cv2.putText(frame, text, (text_x, text_y), font, scale, color, thickness)
        return frame
    
    # ADDED: Method to force reset all states
    def force_reset(self):
        """Force reset all reader states"""
        self.counting = False
        self.count_val = 5


# Initialize reader
if "reader" not in st.session_state:
    st.session_state.reader = PalmReader()


reader = st.session_state.reader


# ADDED: Clean reset function
def clean_reset_states():
    """Clean reset of all states to ensure fresh start"""
    # Reset session states
    st.session_state.analysis_result = None
    st.session_state.captured_image = None
    st.session_state.analyzing = False
    st.session_state.show_qr = False
    st.session_state.show_results_directly = False
    st.session_state.analysis_time = 0
    st.session_state.banner = None
    
    # Reset reader states
    reader.force_reset()
    
    # Ensure camera stays active - CRITICAL FIX
    # Don't reset camera_active to maintain camera connection
    print("🔄 States cleaned - camera remains active")


# ── Ultra-fast threading function ─────────────────────────────────────────
def start_lightning_countdown():
    """Ultra-fast countdown and analysis with 10-second guarantee"""
    
    def lightning_countdown_and_capture():
        """Lightning countdown with absolute speed guarantee"""
        reader.counting = True
        analysis_start_time = None
        
        try:
            # FIXED: Corrected countdown range to match 5 seconds
            for i in range(5, 0, -1):  # CHANGED: Now actually 5 seconds as intended
                reader.count_val = i
                st.session_state.banner = str(i)
                time.sleep(1)
            
            # Capture
            st.session_state.banner = "Captured!"
            reader.count_val = 0
            time.sleep(0.3)  # Shorter delay
            
            # Get frame
            if reader.cap and reader.cap.isOpened():
                ret, frame = reader.cap.read()
                if ret:
                    # Convert to PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    captured_img = Image.fromarray(frame_rgb)
                    
                    # Store captured image
                    st.session_state.captured_image = captured_img
                    st.session_state.analyzing = True
                    st.session_state.show_results_directly = True
                    
                    # Lightning analysis with timing
                    analysis_start_time = time.time()
                    print("⚡ Starting analysis...")
                    
                    # Call ultra-fast analysis
                    analysis, elapsed_time = analyze_palm_lightning(captured_img)
                    
                    # Store results with timing
                    st.session_state.analysis_result = analysis
                    st.session_state.analysis_time = elapsed_time
                    st.session_state.analyzing = False
                    
                    # Complete
                    st.session_state.banner = None
                    st.session_state.show_qr = False
                    reader.counting = False  # IMPORTANT: Reset counting state
                    
                    print(f"🎯 Complete analysis pipeline finished in {elapsed_time:.2f}s")
                    
                else:
                    print("❌ Failed to capture frame")
                    st.session_state.analysis_result = generate_instant_fallback()
                    st.session_state.analysis_time = 0.1
                    st.session_state.analyzing = False
                    st.session_state.banner = None
                    reader.counting = False  # IMPORTANT: Reset counting state
            
        except Exception as e:
            print(f"❌ Thread error: {str(e)}")
            
            # Always provide instant fallback on any error
            st.session_state.analysis_result = generate_instant_fallback()
            st.session_state.analysis_time = 0.1 if analysis_start_time is None else time.time() - analysis_start_time
            st.session_state.analyzing = False
            st.session_state.banner = None
            reader.counting = False  # IMPORTANT: Reset counting state
        
    # Create lightning-fast thread
    lightning_thread = threading.Thread(target=lightning_countdown_and_capture, daemon=True)
    add_script_run_ctx(lightning_thread)
    lightning_thread.start()


# ── Main App Logic ─────────────────────────────────────────────────────────
def show_main_page():
    """Main page with ultra-fast results"""
    st.title("⚡ AI Tech Fortune Reader")
    
    # Banner for countdown
    if st.session_state.banner:
        st.markdown(
            f"<h1 style='text-align:center;color:#FFD700;font-size:72px;'>{st.session_state.banner}</h1>",
            unsafe_allow_html=True
        )
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.header("⚡ Controls")
        
        # ENHANCED: Camera controls with better state management
        if not st.session_state.camera_active:
            if st.button("📹 Initialize Camera", type="primary", key="init_camera"):
                if reader.cam_on():
                    st.session_state.camera_active = True
                    st.success("✅ Camera initialized!")
                    st.rerun()
                else:
                    st.error("❌ Failed to initialize camera")
        else:
            # ADDED: Show camera status and option to reinitialize
            st.success("✅ Camera Active")
            if st.button("🔄 Reinitialize Camera", key="reinit_camera"):
                reader.cam_off()
                time.sleep(0.5)  # Brief pause
                if reader.cam_on():
                    st.session_state.camera_active = True
                    st.success("✅ Camera reinitialized!")
                    st.rerun()
                else:
                    st.error("❌ Failed to reinitialize camera")
                    st.session_state.camera_active = False
                    st.rerun()
        
        st.divider()
        
        # FIXED: Enhanced button state logic
        start_reading_disabled = (
            not st.session_state.camera_active or 
            reader.counting or 
            st.session_state.analyzing or
            not (reader.cap and reader.cap.isOpened())  # ADDED: Check camera is actually working
        )
        
        # ADDED: Debug info for troubleshooting
        with st.expander("🔧 Debug Info", expanded=False):
            st.write(f"Camera Active: {st.session_state.camera_active}")
            st.write(f"Reader Counting: {reader.counting}")
            st.write(f"Analyzing: {st.session_state.analyzing}")
            st.write(f"Camera Object OK: {reader.cap and reader.cap.isOpened()}")
            st.write(f"Button Disabled: {start_reading_disabled}")
        
        if st.button("⚡ Tech Reading", 
                    disabled=start_reading_disabled,
                    type="secondary", 
                    key="start_reading"):
            start_lightning_countdown()
        
        # ENHANCED: Show more detailed status
        if reader.counting:
            st.info("🔄 Countdown (5 seconds)...")
        elif st.session_state.analyzing:
            st.info("⚡ Ultra-fast analysis in progress... (max 10s)")
            
            # Real-time speed indicator
            start_time = time.time()
            speed_placeholder = st.empty()
            
            # Show live timer
            for i in range(100):  # 10 seconds max
                if not st.session_state.analyzing:
                    break
                elapsed = time.time() - start_time
                remaining = max(0, 10 - elapsed)
                speed_placeholder.markdown(f"⏱️ **Time remaining:** {remaining:.1f}s")
                time.sleep(0.1)
            
            speed_placeholder.empty()
        elif start_reading_disabled and st.session_state.camera_active:
            st.warning("⚠️ Camera check failed - try reinitializing")
        
        # ENHANCED: Reset button with better state management
        if st.session_state.show_results_directly:
            st.divider()
            if st.button("🔄 New Reading", type="primary", key="reset_main"):
                clean_reset_states()  # USES NEW CLEAN RESET FUNCTION
                st.rerun()
        
        st.divider()
        
        # Lightning instructions
        st.subheader("⚡ Instructions")
        st.markdown("""
        **Step 1:** Initialize Camera (instant)
        
        **Step 2:** Position your palm clearly
        
        **Step 3:** Click "Tech Reading"
        
        **Troubleshooting:** If button stays disabled, try "Reinitialize Camera"
        """)
    
    with col1:
        st.header("📸 Camera")
        
        if st.session_state.camera_active and reader.cap and reader.cap.isOpened():
            # Create placeholder for video feed
            video_placeholder = st.empty()
            
            # Optimized loop with faster frame rate
            frame_count = 0
            max_frames = 500  # Reduced for speed
            
            while (st.session_state.camera_active and 
                   frame_count < max_frames and
                   not st.session_state.show_results_directly):
                
                ret, frame = reader.cap.read()
                if not ret:
                    st.error("❌ Failed to read from camera")
                    break
                
                # FIXED: Add overlay if counting - Remove problematic emoji
                if reader.counting:
                    if reader.count_val > 0:
                        frame = reader.overlay_text(frame, str(reader.count_val), (0, 255, 0))  # CLEAN NUMBER ONLY
                    else:
                        frame = reader.overlay_text(frame, "Captured!", (255, 255, 0))  # SIMPLE TEXT
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                frame_count += 1
                time.sleep(0.02)  # Faster frame rate
        else:
            st.info("📹 Click 'Initialize Camera' to start video feed")
    
    # Show results directly below
    if st.session_state.show_results_directly:
        st.divider()
        show_lightning_results()


def show_lightning_results():
    """Show ultra-fast results section with side-by-side layout"""
    st.header("⚡ Your Tech Fortune!")
    
    if st.session_state.captured_image:
        # CHANGED: Create side-by-side layout for image and results
        col_img, col_results = st.columns([1, 2])  # Image gets 1/3, results get 2/3 of width
        
        with col_img:
            # CHANGED: Smaller image with fixed width
            st.image(st.session_state.captured_image, 
                    caption="Analysis source", 
                    width=300)  # Fixed width instead of use_container_width=True
            
            # Speed stats moved here
            if st.session_state.analysis_time > 0:
                st.markdown("**⚡ Speed:**")
                st.metric("Time", f"{st.session_state.analysis_time:.2f}s", 
                         delta=f"{10 - st.session_state.analysis_time:.1f}s under limit")
                
                if st.session_state.analysis_time < 3:
                    st.success("🚀 Ultra-Fast!")
                elif st.session_state.analysis_time < 7:
                    st.info("⚡ Lightning!")
                else:
                    st.warning("🔄 Fallback")
        
        with col_results:
            # CHANGED: Results now appear alongside the image
            if st.session_state.analyzing:
                st.error("⚠️ Analysis timeout - delivering instant backup!")
                st.session_state.analysis_result = generate_instant_fallback()
                st.session_state.analysis_time = 0.1
                st.session_state.analyzing = False
                st.rerun()
                    
            elif st.session_state.analysis_result:
                if not st.session_state.show_qr:
                    # Show results with speed indicator
                    if st.session_state.analysis_time < 3:
                        st.subheader("🚀 Your Ultra-Fast Tech Fortune")
                    elif st.session_state.analysis_time < 7:
                        st.subheader("⚡ Your Lightning Tech Fortune")
                    else:
                        st.subheader("🔄 Your Instant Backup Fortune")
                    
                    # Display results in the right column
                    st.markdown(st.session_state.analysis_result)
                    
                    # Speed achievement
                    if st.session_state.analysis_time < 5:
                        st.balloons()  # Celebrate fast results!
                
                else:
                    # Show sharing options in right column
                    st.subheader("📱 Share Your Lightning Fortune")
                    
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Add timing info to the share text
                    analysis_text_with_timing = st.session_state.analysis_result + f"\n\n⚡ *Generated in just {st.session_state.analysis_time:.2f} seconds!*"
                    
                    # Create compact reading
                    compact_reading = create_compact_reading(analysis_text_with_timing, timestamp)
                    
                    # Generate QR code
                    try:
                        qr_code_base64 = generate_qr_code(compact_reading)
                    except Exception as e:
                        st.error(f"QR Code generation error: {str(e)}")
                        qr_code_base64 = None
                    
                    # QR Code display
                    st.markdown("### 📱 Lightning QR Code")
                    if qr_code_base64:
                        st.markdown(
                            f'<div style="text-align: center;"><img src="data:image/png;base64,{qr_code_base64}" style="width: 100%; max-width: 200px; border: 3px solid #FFD700; border-radius: 15px; padding: 10px; background: white;"></div>',
                            unsafe_allow_html=True
                        )
                        st.success("📱 **Scan for instant copy!**")
                    else:
                        st.error("❌ Could not generate QR code")
                    
                    # Download section
                    st.markdown("### 💾 Download Report")
                    
                    # Create downloadable HTML
                    html_content = create_downloadable_html(analysis_text_with_timing, timestamp)
                    
                    # Download button
                    st.download_button(
                        label="⚡ Download Lightning Report",
                        data=html_content,
                        file_name=f"lightning_fortune_{timestamp.replace(':', '-').replace(' ', '_')}.html",
                        mime="text/html",
                        use_container_width=True,
                        key="download_report"
                    )
            
            else:
                st.error("❌ No results available - this shouldn't happen with Lightning Mode!")
                if st.button("⚡ Generate Instant Fortune", key="generate_instant_fallback"):
                    st.session_state.analysis_result = generate_instant_fallback()
                    st.session_state.analysis_time = 0.1
                    st.rerun()
        
        # Action buttons below both columns
        st.divider()
        
        # FIXED: Action buttons with clean reset
        if st.session_state.analysis_result and not st.session_state.show_qr:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("⚡ New Lightning Reading", type="primary", use_container_width=True, key="new_reading_main"):
                    clean_reset_states()  # USES NEW CLEAN RESET FUNCTION
                    st.rerun()
            
            with col2:
                if st.button("📤 Share Lightning Results", type="secondary", use_container_width=True, key="share_results_main"):
                    st.session_state.show_qr = True
                    st.rerun()
        
        elif st.session_state.show_qr:
            # FIXED: Back buttons for QR mode with clean reset
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("⬅️ Back to Results", use_container_width=True, key="back_to_results_qr"):
                    st.session_state.show_qr = False
                    st.rerun()
            
            with col2:
                if st.button("⚡ New Lightning Reading", type="primary", use_container_width=True, key="new_reading_qr"):
                    clean_reset_states()  # USES NEW CLEAN RESET FUNCTION
                    st.rerun()


def main():
    """Main application with lightning theming"""
    
    # Lightning CSS styling
    st.markdown("""
    <style>
    .lightning-header {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    .speed-guarantee {
        background: rgba(255, 215, 0, 0.2);
        border: 2px solid #FFD700;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .lightning-border {
        border: 3px solid #FFD700;
        border-radius: 15px;
        padding: 20px;
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(255, 165, 0, 0.1));
    }
    
    .speed-metric {
        background: rgba(0, 255, 0, 0.1);
        border-left: 5px solid #00FF00;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #FFD700, #FFA500);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Show lightning main page
    show_main_page()


# ── Run Lightning App ──────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
