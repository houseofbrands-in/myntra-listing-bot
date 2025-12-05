import streamlit as st
import pandas as pd
import json
import base64
import requests
from openai import OpenAI
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from io import BytesIO
import zipfile
from PIL import Image, ImageOps
import google.generativeai as genai
import difflib 

# --- DEPENDENCY CHECK ---
try:
    from rembg import remove as remove_bg_ai
    REMBG_AVAILABLE = True
except ImportError as e:
    REMBG_AVAILABLE = False

st.set_page_config(page_title="HOB OS - V10.6", layout="wide")

# ==========================================
# 1. AUTHENTICATION & DATABASE CONNECT
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_role = ""
    st.session_state.username = ""

# --- CACHED CONNECTION ---
@st.cache_resource
def init_connection():
    """Establish connection to Google Sheets once."""
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        return None

# --- CACHED DATA FETCHING ---
@st.cache_data
def get_worksheet_data(sheet_name, worksheet_name):
    """Fetch all records from a worksheet and cache them."""
    client = init_connection()
    if not client: return []
    try:
        sh = client.open(sheet_name)
        ws = sh.worksheet(worksheet_name)
        return ws.get_all_values()
    except: return []

# Global Constants
SHEET_NAME = "Agency_OS_Database"

# --- API SETUP ---
try:
    # OpenAI
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
    
    # Gemini
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            GEMINI_AVAILABLE = True
        else:
            GEMINI_AVAILABLE = False
    except:
        GEMINI_AVAILABLE = False

except Exception as e:
    st.error(f"❌ Secrets Error: {str(e)}")
    st.stop()

# ==========================================
# 2. CORE LOGIC & UTILS
# ==========================================

def get_worksheet_object(ws_name):
    """Helper to get write-access worksheet object"""
    gc = init_connection()
    return gc.open(SHEET_NAME).worksheet(ws_name)

def check_login(username, password):
    rows = get_worksheet_data(SHEET_NAME, "Users")
    if not rows: return False, None
    for row in rows[1:]: 
        if len(row) >= 3:
            if str(row[0]).strip() == username and str(row[1]).strip() == password:
                return True, row[2]
    return False, None

def create_user(username, password, role):
    try:
        ws = get_worksheet_object("Users")
        existing = [r[0] for r in get_worksheet_data(SHEET_NAME, "Users")]
        if username in existing: return False, "User exists"
        ws.append_row([username, password, role])
        st.cache_data.clear()
        return True, "Created"
    except Exception as e: return False, str(e)

def delete_user(username):
    try:
        ws = get_worksheet_object("Users")
        cell = ws.find(username)
        if cell: 
            ws.delete_rows(cell.row)
            st.cache_data.clear()
            return True
        return False
    except: return False

def get_all_users(): 
    rows = get_worksheet_data(SHEET_NAME, "Users")
    if len(rows) > 1:
        headers = rows[0]
        return [dict(zip(headers, r)) for r in rows[1:]]
    return []

def get_categories_for_marketplace(marketplace):
    rows = get_worksheet_data(SHEET_NAME, "Configs")
    cats = [row[1] for row in rows if len(row) > 1 and row[0] == marketplace]
    return list(set([c for c in cats if c and c != "Category"]))

def save_config(marketplace, category, data):
    try:
        ws = get_worksheet_object("Configs")
        json_str = json.dumps(data)
        all_vals = ws.get_all_values()
        for i, row in enumerate(all_vals):
            if len(row) > 1 and row[0] == marketplace and row[1] == category:
                ws.update_cell(i + 1, 3, json_str)
                st.cache_data.clear()
                return True
        ws.append_row([marketplace, category, json_str])
        st.cache_data.clear()
        return True
    except: return False

def load_config(marketplace, category):
    rows = get_worksheet_data(SHEET_NAME, "Configs")
    for row in rows:
        if len(row) > 2 and row[0] == marketplace and row[1] == category:
            return json.loads(row[2])
    return None

def delete_config(marketplace, category):
    try:
        ws = get_worksheet_object("Configs")
        all_vals = ws.get_all_values()
        for i, row in enumerate(all_vals):
            if len(row) > 1 and row[0] == marketplace and row[1] == category:
                ws.delete_rows(i + 1)
                st.cache_data.clear()
                return True
        return False
    except: return False

def save_seo(marketplace, category, keywords_list):
    try:
        ws = get_worksheet_object("SEO_Data")
        kw_string = ", ".join([str(k).strip() for k in keywords_list if str(k).strip()])
        all_vals = ws.get_all_values()
        for i, row in enumerate(all_vals):
            if len(row) > 1 and row[0] == marketplace and row[1] == category:
                ws.update_cell(i + 1, 3, kw_string)
                st.cache_data.clear()
                return True
        ws.append_row([marketplace, category, kw_string])
        st.cache_data.clear()
        return True
    except: return False

def get_seo(marketplace, category):
    rows = get_worksheet_data(SHEET_NAME, "SEO_Data")
    for row in rows:
        if len(row) > 2 and row[0] == marketplace and row[1] == category:
            return row[2]
    return ""

def parse_master_data(file):
    df = pd.read_excel(file)
    valid_options = {}
    for col in df.columns:
        options = df[col].dropna().astype(str).unique().tolist()
        if len(options) > 0: valid_options[col] = options
    return valid_options

# --- ROBUST IMAGE DOWNLOADER ---
def encode_image_from_url(url):
    try:
        if pd.isna(url) or str(url).strip() == "": 
            return None, "Empty URL"
        
        url = str(url).strip()
        
        # Dropbox Fix
        if "dropbox.com" in url:
            url = url.replace("?dl=0", "").replace("&dl=0", "") + ("&dl=1" if "?" in url else "?dl=1")
        
        # Google Drive Fix
        if "drive.google.com" in url and "/view" in url:
            file_id = url.split("/d/")[1].split("/")[0]
            url = f"https://drive.google.com/uc?export=download&id={file_id}"

        # --- USER AGENT FIX ---
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Referer": "https://www.google.com/"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8'), None
        else:
            return None, f"Download Error: Status {response.status_code}"
            
    except Exception as e: 
        return None, f"Network Error: {str(e)}"

# --- MASTER DATA ENFORCER (STRICT) ---
def enforce_master_data(value, options):
    """Forces AI value to closest match in Master Data list."""
    if not value: return ""
    str_val = str(value).strip()
    
    # 1. Exact Match
    for opt in options:
        if str(opt).lower() == str_val.lower():
            return opt 
            
    # 2. Fuzzy Match
    matches = difflib.get_close_matches(str_val, [str(o) for o in options], n=1, cutoff=0.6)
    if matches:
        return matches[0]
        
    return ""

# --- SMART TRUNCATOR (Layer 2 Defense) ---
def smart_truncate(text, max_length):
    """
    Cuts text to limit without splitting words.
    Example: "Soft Cotton Fab..." -> "Soft Cotton"
    """
    if not text: return ""
    text = str(text).strip()
    
    # If already within limit, return as is
    if len(text) <= max_length:
        return text
    
    # Hard cut to limit
    truncated = text[:max_length]
    
    # If the character AFTER our cut was not a space, we likely cut a word in half.
    # So we backtrack to the last space in our truncated string.
    if len(text) > max_length and text[max_length] != " ":
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
    
    return truncated.strip()

# --- IMAGE PROCESSOR (TOOLS) ---
def process_image_advanced(image_file, target_w, target_h, mode, do_remove_bg):
    try:
        img = Image.open(image_file)
        if do_remove_bg:
            if REMBG_AVAILABLE: img = remove_bg_ai(img)
            else: return None, "rembg not installed"
        img = img.convert("RGBA")

        if mode == "Stretch to Target (Distort)":
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            final_bg = Image.new("RGB", (target_w, target_h), (255, 255, 255))
            final_bg.paste(img, (0, 0), img)
            return final_bg, None

        elif mode == "Resize Only (No Padding)":
            img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
            final_w, final_h = img.size
            final_bg = Image.new("RGB", (final_w, final_h), (255, 255, 255))
            final_bg.paste(img, (0, 0), img)
            return final_bg, None

        elif mode == "Scale & Pad (White Bars)":
            img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
            final_bg = Image.new("RGB", (target_w, target_h), (255, 255, 255))
            left = (target_w - img.width) // 2
            top = (target_h - img.height) // 2
            final_bg.paste(img, (left, top), img)
            return final_bg, None
    except Exception as e: return None, str(e)

# --- VISION GUARD (COMPLIANCE CHECKER) ---
def validate_image_compliance(image_file, marketplace):
    try:
        img = Image.open(image_file)
        w, h = img.size
        ratio = w / h
        status = "✅ PASS"
        issues = []

        # 1. Resolution Check
        if marketplace == "Amazon":
            if w < 1000 or h < 1000:
                issues.append(f"Low Res ({w}x{h} < 1000px)")
            if not (0.95 <= ratio <= 1.05):
                issues.append(f"Not Square (Ratio: {ratio:.2f})")
        
        elif marketplace == "Myntra":
            if h < 900:
                issues.append(f"Low Height ({h} < 900px)")
            if not (0.65 <= ratio <= 0.85): # Approx 3:4
                issues.append(f"Wrong Aspect Ratio ({ratio:.2f})")

        # 2. Background Check (Simple Corner Check)
        if marketplace == "Amazon":
            img_rgb = img.convert("RGB")
            corners = [img_rgb.getpixel((0, 0)), img_rgb.getpixel((w-1, 0))]
            is_white = all(sum(c) > 750 for c in corners) # >750 means very close to 255,255,255
            if not is_white:
                issues.append("Background not White")

        if issues:
            status = "❌ FAIL"
        
        return {"Filename": image_file.name, "Status": status, "Issues": ", ".join(issues), "Dimensions": f"{w}x{h}"}

    except Exception as e:
        return {"Filename": image_file.name, "Status": "⚠️ ERROR", "Issues": str(e), "Dimensions": "N/A"}

# --- AI LOGIC (GPT + GEMINI) ---
def analyze_image_hybrid(model_choice, client, image_url, user_hints, keywords, config, marketplace):
    # 1. Prepare Image
    base64_image, error = encode_image_from_url(image_url)
    if error: return None, error

    # 2. Sort columns & Inject Limits
    tech_cols = []
    creative_cols = []
    gemini_constraints = [] 
    relevant_options = {} 
    
    # Prepare detailed target list with limits for the Prompt
    target_definitions = []

    for col, settings in config['column_mapping'].items():
        # Get Max Len if it exists
        max_len = settings.get('max_len', '')
        limit_txt = f" (MAX LENGTH: {max_len} chars)" if max_len and str(max_len).isdigit() else ""
        
        if settings['source'] == 'AI':
            # Check for Master Data
            master_options = []
            for master_col, opts in config['master_data'].items():
                if master_col.lower() in col.lower() or col.lower() in master_col.lower():
                    master_options = opts
                    break
            
            if master_options:
                tech_cols.append(col)
                relevant_options[col] = master_options
                gemini_constraints.append(f"- Column '{col}': MUST be one of {json.dumps(master_options)}")
                target_definitions.append(f"{col}") # No limit needed usually for dropdowns
            else:
                creative_cols.append(col)
                target_definitions.append(f"{col}{limit_txt}")

    # 3. Marketplace Rules
    seo_section = f"SEO KEYWORDS: {keywords}" if keywords else ""
    mp_rules = ""
    if marketplace.lower() == "amazon":
        mp_rules = "- Bullet Points: 5 bullets. START with BOLD header. - Title: Brand + Dept + Material + Pattern + Style."
    elif marketplace.lower() == "myntra":
        mp_rules = "- Title: Short, punchy, Brand + Category + Style."

    # 4. ENGINE SWITCHING
    try:
        # ======================================================
        # OPTION A: GPT-4o
        # ======================================================
        if "GPT" in model_choice:
            prompt = f"""
            You are a Data Expert for {marketplace}.
            TASK: Generate JSON for these columns: {target_definitions}
            CONTEXT: {user_hints} {seo_section} {mp_rules}
            STRICT DATA RULES: {json.dumps(relevant_options)}
            IMPORTANT: Adhere strictly to MAX LENGTH limits if specified.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a JSON-only assistant."},
                    {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
                ],
                response_format={"type": "json_object"},
                max_tokens=1500
            )
            return json.loads(response.choices[0].message.content), None

        # ======================================================
        # OPTION B: GEMINI 2.5 FLASH
        # ======================================================
        elif "Gemini" in model_choice:
            if not GEMINI_AVAILABLE: return None, "Gemini API Key missing."
            
            model = genai.GenerativeModel('gemini-2.5-flash') 
            img_data = base64.b64decode(base64_image)
            image_part = {"mime_type": "image/jpeg", "data": img_data}
            
            gemini_prompt = f"""
            Cataloging Bot for {marketplace}.
            
            TECHNICAL COLUMNS (STRICT SELECTION):
            {chr(10).join(gemini_constraints)}
            
            CREATIVE COLUMNS (OUTPUT JSON):
            Targets: {target_definitions}
            Rules: {mp_rules} {seo_section}
            """
            
            response = model.generate_content([gemini_prompt, image_part])
            text_out = response.text
            if "```json" in text_out: text_out = text_out.split("```json")[1].split("```")[0]
            elif "```" in text_out: text_out = text_out.split("```")[1].split("```")[0]
            return json.loads(text_out), None

    except Exception as e: return None, str(e)
    return None, "Unknown Error"

# ==========================================
# 3. MAIN APP
# ==========================================

if not st.session_state.logged_in:
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.title("🔒 HOB OS Login")
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                is_valid, role = check_login(user, pwd)
                if is_valid:
                    st.session_state.logged_in = True; st.session_state.username = user; st.session_state.user_role = role
                    st.rerun()
                else: st.error("Invalid Credentials")
else:
    # --- SIDEBAR ---
    st.sidebar.title("🌍 HOB OS")
    st.sidebar.caption(f"User: {st.session_state.username} | Role: {st.session_state.user_role}")
    if st.sidebar.button("Log Out"): st.session_state.logged_in = False; st.rerun()
    st.sidebar.divider()

    selected_mp = st.sidebar.selectbox("Marketplace", ["Myntra", "Flipkart", "Ajio", "Amazon", "Nykaa"])
    if st.sidebar.button("🔄 Refresh Data"): st.rerun()
    
    with st.sidebar.expander("ℹ️ Guide"):
        st.markdown("**1. Setup:** Map Columns.\n**2. Run:** Generate Listing.\n**3. Tools:** Process Images.")

    mp_cats = get_categories_for_marketplace(selected_mp)
    
    base_tabs = ["🛠️ Setup", "📈 SEO", "🚀 Run", "🖼️ Tools"]
    if st.session_state.user_role.lower() == "admin": base_tabs += ["🗑️ Configs", "👥 Admin"]
    tabs = st.tabs(base_tabs)

    # --- TAB 1: SETUP ---
    with tabs[0]:
        st.header(f"1. Setup {selected_mp} Rules")
        mode = st.radio("Mode", ["Create New", "Edit Existing"], horizontal=True)
        cat_name = ""; headers = []; master_options = {}; default_mapping = []

        if mode == "Edit Existing":
            if mp_cats:
                edit_cat = st.selectbox(f"Select Category", mp_cats)
                if edit_cat:
                    loaded = load_config(selected_mp, edit_cat)
                    if loaded:
                        cat_name = loaded['category_name']; headers = loaded['headers']; master_options = loaded['master_data']
        else: cat_name = st.text_input(f"New {selected_mp} Category Name")

        c1, c2 = st.columns(2)
        template_file = c1.file_uploader("Template (.xlsx)", type=["xlsx"])
        master_file = c2.file_uploader("Master Data (.xlsx)", type=["xlsx"])

        if template_file: headers = pd.read_excel(template_file).columns.tolist()
        if master_file: master_options = parse_master_data(master_file)

        if headers:
            st.divider()
            
            # Construct DataFrame for Editor
            if not default_mapping:
                for h in headers:
                    src = "Leave Blank"; h_low = h.lower()
                    if "image" in h_low or "sku" in h_low: src = "Input Excel"
                    elif h in master_options or "name" in h_low or "desc" in h_low: src = "AI Generation"
                    # Default structure now includes Max Len
                    default_mapping.append({
                        "Column Name": h, 
                        "Source": src, 
                        "Fixed Value": "", 
                        "Max Chars": "" 
                    })
            
            # Create a UI-friendly list from the loaded config if it exists
            ui_data = []
            if mode == "Edit Existing" and loaded:
                for col, rule in loaded['column_mapping'].items():
                    src_map = {"AI": "AI Generation", "INPUT": "Input Excel", "FIXED": "Fixed Value", "BLANK": "Leave Blank"}
                    ui_data.append({
                        "Column Name": col,
                        "Source": src_map.get(rule['source'], "Leave Blank"),
                        "Fixed Value": rule.get('value', ''),
                        "Max Chars": rule.get('max_len', '') 
                    })
            else:
                ui_data = default_mapping

            # Configure Columns
            edited_df = st.data_editor(
                pd.DataFrame(ui_data),
                column_config={
                    "Source": st.column_config.SelectboxColumn("Source", options=["Input Excel", "AI Generation", "Fixed Value", "Leave Blank"], width="medium"),
                    "Fixed Value": st.column_config.TextColumn("Fixed Value", width="medium"),
                    "Max Chars": st.column_config.NumberColumn("Max Chars", min_value=1, max_value=5000, step=1, help="Leave 0 or blank for no limit.", width="small")
                },
                hide_index=True,
                use_container_width=True,
                height=400
            )
            
            if st.button("Save Config"):
                final_map = {}
                for i, row in edited_df.iterrows():
                    src_code = "AI" if row['Source'] == "AI Generation" else "INPUT" if row['Source'] == "Input Excel" else "FIXED" if row['Source'] == "Fixed Value" else "BLANK"
                    
                    # Handle Max Len
                    m_len = row['Max Chars']
                    if pd.isna(m_len) or str(m_len).strip() == "" or str(m_len) == "0":
                        m_len = ""
                    else:
                        m_len = int(m_len)

                    final_map[row['Column Name']] = {
                        "source": src_code, 
                        "value": row['Fixed Value'],
                        "max_len": m_len
                    }
                
                if save_config(selected_mp, cat_name, {"category_name": cat_name, "headers": headers, "master_data": master_options, "column_mapping": final_map}):
                    st.success("Saved!"); time.sleep(1); st.rerun()

    # --- TAB 2: SEO ---
    with tabs[1]:
        st.header(f"2. SEO Keywords")
        if not mp_cats: st.warning("Create category first.")
        else:
            seo_cat = st.selectbox("Category", mp_cats, key="seo")
            curr_kw = get_seo(selected_mp, seo_cat)
            if curr_kw: st.info(f"Active: {curr_kw[:60]}...")
            kw_file = st.file_uploader("Keywords File (Col A)", type=["xlsx", "csv"])
            if kw_file and st.button("Update"):
                df_kw = pd.read_csv(kw_file) if kw_file.name.endswith('.csv') else pd.read_excel(kw_file)
                if save_seo(selected_mp, seo_cat, df_kw.iloc[:, 0].dropna().astype(str).tolist()):
                    st.success("Updated!"); time.sleep(1); st.rerun()

    # --- TAB 3: RUN (V10.6 INTEGRATED) ---
    with tabs[2]:
        st.header(f"3. Run {selected_mp} Generator")
        
        if not mp_cats: 
            st.warning("No categories configured yet.")
            st.stop()
        
        run_cat = st.selectbox("Select Category", mp_cats, key="run")
        config = None
        if run_cat:
            config = load_config(selected_mp, run_cat)
            if config:
                required_cols = ["Image URL"] + [col for col, rule in config.get('column_mapping', {}).items() if rule.get('source') == 'INPUT']
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer: 
                    pd.DataFrame(columns=required_cols).to_excel(writer, index=False)
                st.download_button("📥 Download Input Template", output.getvalue(), file_name=f"{selected_mp}_{run_cat}_Template.xlsx")

        active_kws = get_seo(selected_mp, run_cat)
        input_file = st.file_uploader("Upload Input Data (filled template)", type=["xlsx"], key="run_in")
        
        if input_file:
            df_input = pd.read_excel(input_file)
            total_rows = len(df_input)
            
            # --- MODEL & COLUMN SELECTION ---
            st.divider()
            c_run1, c_run2, c_run3 = st.columns([1, 1, 1])
            with c_run1:
                run_mode = st.radio("Processing Mode", ["🧪 Test Run (First 3 Rows)", "🚀 Full Production Run"])
            with c_run2:
                model_select = st.selectbox("AI Model Engine", ["GPT-4o", "Gemini 2.5 Flash"])
            with c_run3:
                # Manual Column Selector
                potential_cols = [c for c in df_input.columns if "image" in c.lower() or "url" in c.lower() or "link" in c.lower()]
                default_idx = df_input.columns.get_loc(potential_cols[0]) if potential_cols else 0
                img_col = st.selectbox("Select Image URL Column", df_input.columns, index=default_idx)

            if run_mode.startswith("🧪"):
                df_to_process = df_input.head(3)
                st.info("Test Mode: Processing first 3 rows only.")
            else:
                df_to_process = df_input
                st.warning(f"Production Mode: Processing {total_rows} rows.")

            est_cost = len(df_to_process) * 0.02
            st.metric("Estimated Cost", f"${est_cost:.2f}")

            if st.button("▶️ Start Generation"):
                progress = st.progress(0)
                status = st.empty()
                error_log = st.empty() 
                final_rows = []
                cache = {}
                
                if not config: st.error("Config not loaded"); st.stop()
                mapping = config['column_mapping']
                current_total = len(df_to_process)
                
                # --- PROCESSING LOOP ---
                for idx, row in df_to_process.iterrows():                  
                    # --- SAFETY BRAKE FOR GEMINI ---
                    if "Gemini" in model_select:
                        time.sleep(5) 

                    status.text(f"Processing Row {idx+1}/{current_total} ({model_select})...")
                    progress.progress((idx+1)/current_total)
                    
                    img_url = str(row.get(img_col, "")).strip()
                    st.caption(f"🔎 Row {idx+1} URL: {img_url}")

                    ai_data = None
                    last_error = "Unknown Error"
                    
                    needs_ai = any(m['source']=='AI' for m in mapping.values())
                    
                    if needs_ai:
                        if not img_url or img_url.lower() == "nan":
                            error_log.warning(f"⚠️ Row {idx+1}: Image URL is empty.")
                        elif img_url in cache: 
                            ai_data = cache[img_url]
                        else:
                            hints = ", ".join([f"{k}: {v}" for k,v in row.items() if str(v) != "nan" and k != img_col])
                            
                            attempts = 0
                            max_retries = 2
                            while attempts < max_retries:
                                ai_data, last_error = analyze_image_hybrid(model_select, client, img_url, hints, active_kws, config, selected_mp)
                                if ai_data: 
                                    break 
                                else:
                                    attempts += 1
                                    time.sleep(1) 
                            
                            if ai_data: 
                                cache[img_url] = ai_data
                            else:
                                error_log.error(f"❌ FAILED [Row {idx+1}]: {last_error}")
                    
                    # --- MAPPING & ENFORCEMENT ---
                    new_row = {}
                    for col in config['headers']:
                        rule = mapping.get(col, {'source': 'BLANK'})
                        final_val = ""

                        if rule['source'] == 'INPUT':
                            if col in df_input.columns: final_val = row[col]
                            else:
                                for ic in df_input.columns:
                                    if ic.lower() in col.lower(): final_val = row[ic]; break

                        elif rule['source'] == 'FIXED': 
                            final_val = rule['value']

                        elif rule['source'] == 'AI':
                            if ai_data:
                                if col in ai_data: 
                                    final_val = ai_data[col]
                                else:
                                    clean_col = col.lower().replace(" ", "").replace("_", "")
                                    for k, v in ai_data.items():
                                        clean_k = k.lower().replace(" ", "").replace("_", "")
                                        if clean_k in clean_col or clean_col in clean_k:
                                            final_val = v; break
                            
                            # Master Data Enforce
                            master_list = []
                            for master_col, opts in config['master_data'].items():
                                if master_col.lower() in col.lower() or col.lower() in master_col.lower():
                                    master_list = opts
                                    break
                            
                            if master_list and final_val:
                                final_val = enforce_master_data(final_val, master_list)

                        # --- APPLY SMART TRUNCATION (Max Limit) ---
                        max_len = rule.get('max_len')
                        if max_len and str(max_len).isdigit() and int(max_len) > 0:
                            final_val = smart_truncate(final_val, int(max_len))

                        new_row[col] = final_val
                            
                    final_rows.append(new_row)
                
                output_gen = BytesIO()
                with pd.ExcelWriter(output_gen, engine='xlsxwriter') as writer: 
                    pd.DataFrame(final_rows).to_excel(writer, index=False)
                
                st.success("✅ Done!")
                st.download_button("⬇️ Download Result", output_gen.getvalue(), file_name=f"{selected_mp}_{run_cat}_Generated.xlsx")

    # --- TAB 4: TOOLS (VISION GUARD ADDED) ---
    with tabs[3]:
        st.header("🛠️ Media Tools")
        
        tool_mode = st.radio("Select Tool", ["🖼️ Bulk Processor (Resize/BG Removal)", "🛡️ Vision Guard (Compliance Check)"], horizontal=True)
        st.divider()

        if tool_mode == "🖼️ Bulk Processor (Resize/BG Removal)":
            st.subheader("Bulk Image Processor")
            c_tool1, c_tool2 = st.columns(2)
            with c_tool1:
                target_w = st.number_input("Width (px)", value=1000, step=100)
                target_h = st.number_input("Height (px)", value=1000, step=100)
            with c_tool2:
                resize_mode = st.selectbox("Resize Mode", [
                    "Scale & Pad (White Bars)", 
                    "Resize Only (No Padding)", 
                    "Stretch to Target (Distort)"
                ])
                remove_bg = st.checkbox("Remove Background (AI)", help="Required for Amazon Main Image.")
                if remove_bg and not REMBG_AVAILABLE:
                    st.error("❌ 'rembg' library not installed.")

            tool_files = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg", "webp"], accept_multiple_files=True, key="proc_up")
            
            if tool_files and st.button("Process Images"):
                zip_buffer = BytesIO()
                prog_bar = st.progress(0)
                
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for i, f in enumerate(tool_files):
                        processed, err = process_image_advanced(f, target_w, target_h, resize_mode, remove_bg)
                        if processed:
                            img_byte_arr = BytesIO()
                            processed.save(img_byte_arr, format='JPEG', quality=95)
                            fname = f.name.rsplit('.', 1)[0] + "_processed.jpg"
                            zf.writestr(fname, img_byte_arr.getvalue())
                        else: st.warning(f"Failed {f.name}: {err}")
                        prog_bar.progress((i+1)/len(tool_files))
                
                st.success("Complete!")
                st.download_button("⬇️ Download ZIP", zip_buffer.getvalue(), file_name="Processed_Images.zip", mime="application/zip")

        elif tool_mode == "🛡️ Vision Guard (Compliance Check)":
            st.subheader("Marketplace Compliance Audit")
            st.info("Checks images for Resolution, Aspect Ratio, and White Background compliance.")
            
            check_mp = st.selectbox("Target Marketplace", ["Amazon", "Myntra", "Flipkart"])
            audit_files = st.file_uploader("Upload Images to Audit", type=["jpg", "png", "jpeg", "webp"], accept_multiple_files=True, key="audit_up")
            
            if audit_files and st.button("Run Audit"):
                results = []
                for f in audit_files:
                    # Reset file pointer
                    f.seek(0)
                    res = validate_image_compliance(f, check_mp)
                    results.append(res)
                
                df_res = pd.DataFrame(results)
                
                # Visual Feedback
                st.dataframe(df_res.style.map(lambda x: 'color: red' if 'FAIL' in str(x) else 'color: green', subset=['Status']), use_container_width=True)
                
                # Download Report
                csv = df_res.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Audit Report", csv, "vision_guard_report.csv", "text/csv")

    # --- ADMIN TABS ---
    if st.session_state.user_role.lower() == "admin":
        with tabs[4]:
            st.header("Manage Configs")
            to_del = st.selectbox("Delete", [""]+mp_cats)
            if to_del and st.button("Delete Config"):
                delete_config(selected_mp, to_del); st.success("Deleted"); time.sleep(1); st.rerun()

        with tabs[5]:
            st.header("👥 Admin Console")
            st.dataframe(pd.DataFrame(get_all_users()))
            c_add1, c_add2 = st.columns(2)
            with c_add1:
                with st.form("add_user"):
                    new_u = st.text_input("Username"); new_p = st.text_input("Password"); new_r = st.selectbox("Role", ["user", "admin"])
                    if st.form_submit_button("Create"):
                        ok, msg = create_user(new_u, new_p, new_r)
                        if ok: st.success(msg); time.sleep(1); st.rerun()
                        else: st.error(msg)
            with c_add2:
                u_to_del = st.selectbox("Select User", [u['Username'] for u in get_all_users() if str(u['Username']) != "admin"])
                if st.button("Delete"):
                    if delete_user(u_to_del): st.success("Removed"); time.sleep(1); st.rerun()
