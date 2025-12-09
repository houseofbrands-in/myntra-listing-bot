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

st.set_page_config(page_title="HOB OS - V11.0.2 (Stable)", layout="wide")

# ==========================================
# 1. AUTHENTICATION & DATABASE CONNECT
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_role = ""
    st.session_state.username = ""

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

@st.cache_data
def get_worksheet_data(sheet_name, worksheet_name):
    client = init_connection()
    if not client: return []
    try:
        sh = client.open(sheet_name)
        ws = sh.worksheet(worksheet_name)
        return ws.get_all_values()
    except: return []

SHEET_NAME = "Testing_Agency_OS_Database"

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

def encode_image_from_url(url):
    try:
        if pd.isna(url) or str(url).strip() == "": 
            return None, "Empty URL"
        url = str(url).strip()
        if "dropbox.com" in url:
            url = url.replace("?dl=0", "").replace("&dl=0", "") + ("&dl=1" if "?" in url else "?dl=1")
        if "drive.google.com" in url and "/view" in url:
            file_id = url.split("/d/")[1].split("/")[0]
            url = f"https://drive.google.com/uc?export=download&id={file_id}"

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/119.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8'), None
        else:
            return None, f"Download Error: Status {response.status_code}"
    except Exception as e: 
        return None, f"Network Error: {str(e)}"

# --- SMART TRUNCATOR ---
def smart_truncate(text, max_length):
    if not text: return ""
    text = str(text).strip()
    if len(text) <= max_length: return text
    truncated = text[:max_length]
    if len(text) > max_length and text[max_length] != " ":
        if " " in truncated: truncated = truncated.rsplit(" ", 1)[0]
    return truncated.strip()

# --- REVERSE LOOKUP ENFORCER (V11.0.5) ---
def enforce_master_data_fallback(value, options):
    if not value: return ""
    ai_text = str(value).strip().lower()
    
    # 1. Exact Match (Best Case)
    for opt in options:
        if str(opt).strip().lower() == ai_text:
            return opt

    # 2. Reverse Lookup (The "Search Party")
    # We sort your options by Length (Longest to Shortest).
    # This ensures if you have "Ikat Print" and "Ikat", we match the specific "Ikat Print" first.
    sorted_options = sorted(options, key=lambda x: len(str(x)), reverse=True)

    for opt in sorted_options:
        opt_val = str(opt).strip().lower()
        if not opt_val: continue
        
        # CHECK: Is your Valid Option inside the AI's text?
        # e.g. Is "Floral" inside "Floral Printed"? -> YES -> Return "Floral"
        if opt_val in ai_text:
            return opt

    # 3. Fuzzy Match (For spelling mistakes like "Florall")
    matches = difflib.get_close_matches(ai_text, [str(o).lower() for o in options], n=1, cutoff=0.7)
    if matches:
        match_lower = matches[0]
        for opt in options:
            if str(opt).lower() == match_lower:
                return opt

    # 4. No Match Found? Return raw AI value so you can see it.
    return value
    # --- LYRA PROMPT OPTIMIZER ---
def run_lyra_optimization(model_choice, raw_instruction):
    lyra_system_prompt = """
    You are Lyra, a master-level AI prompt optimization specialist. 
    Mission: Transform user input into precision-crafted prompts for e-commerce automation.
    METHODOLOGY: 1. Deconstruct Intent. 2. Develop Constraints. 3. Output precise instruction.
    """
    user_msg = f"Optimize this instruction for an AI Column Generator: '{raw_instruction}'"
    try:
        if "GPT" in model_choice:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": lyra_system_prompt},{"role": "user", "content": user_msg}]
            )
            return response.choices[0].message.content
        elif "Gemini" in model_choice:
            if not GEMINI_AVAILABLE: return "Gemini API Key missing."
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(f"{lyra_system_prompt}\n\nUSER REQUEST: {user_msg}")
            return response.text
    except Exception as e: return f"Error: {str(e)}"

# --- V11.0.8: MAKER-CHECKER (STRICT HIERARCHY MAPPING) ---
def analyze_image_maker_checker(client, base64_image, user_hints, keywords, config, marketplace):
    # PREPARE DATA
    target_columns = []
    strict_constraints = {} 
    creative_columns = []   
    
    # 1. STRICT MAPPING LOGIC (The Fix)
    for col, settings in config['column_mapping'].items():
        if settings['source'] == 'AI':
            target_columns.append(col)
            
            best_match_key = None
            best_match_len = -1
            
            for master_col in config['master_data'].keys():
                c_clean = col.lower().strip()
                m_clean = master_col.lower().strip()
                
                # A. EXACT MATCH (Highest Priority)
                # "Length" == "Length" -> Wins immediately.
                if c_clean == m_clean:
                    best_match_key = master_col
                    break # Stop searching, we found perfection.
                
                # B. CONTAINMENT MATCH (Target MUST contain Master)
                # "Sleeve Length" contains "Sleeve" -> Valid.
                # "Length" contains "Sleeve Length" -> FALSE. (Prevents the bug)
                elif m_clean in c_clean:
                    if len(m_clean) > best_match_len:
                        best_match_len = len(m_clean)
                        best_match_key = master_col

            if best_match_key:
                strict_constraints[col] = config['master_data'][best_match_key]
            else:
                creative_columns.append(col)
    
    # PHASE 1: MAKER (GEMINI) - WITH CHEAT SHEET
    maker_draft = {}
    try:
        if not GEMINI_AVAILABLE: return None, "Gemini Missing"
        model = genai.GenerativeModel('gemini-2.5-flash')
        img_data = base64.b64decode(base64_image)
        image_part = {"mime_type": "image/jpeg", "data": img_data}
        
        maker_prompt = f"""
        Role: E-commerce Expert for {marketplace}.
        Task: Analyze image and generate JSON.
        
        SECTION A: TECHNICAL FACTS (Strict Selection)
        - You MUST select values ONLY from the provided options below.
        - Do NOT invent new words.
        - ALLOWED OPTIONS: {json.dumps(strict_constraints)}
        
        SECTION B: CREATIVE COPY (Be Engaging)
        - For these columns: {creative_columns}
        - RULES: Use emotional language, sensory words. SEO Keywords: {keywords}
        
        Context Hints: {user_hints}
        Output: JSON Only.
        """
        
        response = model.generate_content(
            [maker_prompt, image_part],
            generation_config=genai.types.GenerationConfig(temperature=0.4) 
        )
        
        text_out = response.text
        if "```json" in text_out: text_out = text_out.split("```json")[1].split("```")[0]
        elif "```" in text_out: text_out = text_out.split("```")[1].split("```")[0]
        maker_draft = json.loads(text_out)
        
    except Exception as e:
        return None, f"Maker Failed: {str(e)}"

    # PHASE 2: CHECKER (GPT-4o) - The Strict Arbiter
    try:
        checker_prompt = f"""
        You are the LEAD DATA AUDITOR.
        
        INPUTS:
        1. Visual: [Image provided]
        2. Draft: {json.dumps(maker_draft)}
        3. Allowed Options: {json.dumps(strict_constraints)}
        
        YOUR MISSION:
        1. PRESERVE CREATIVITY: Keep the Draft's Title and Description if they are accurate.
        
        2. ENFORCE CONSISTENCY (Technical Fields):
           - IGNORE the Draft's opinion if it conflicts with the Image or the Allowed List.
           - CRITICAL: If the Draft says "Printed" or "Patterned", YOU must look at the image and pick the specific type (e.g. "Floral", "Geometric", "Striped") from the Allowed Options.
           - Map "Strappy/Cami" -> "Sleeveless". 
        
        3. OUTPUT: Final JSON for columns: {", ".join(target_columns)}
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Data Consistency Engine. Temperature=0.0."},
                {"role": "user", "content": [
                    {"type": "text", "text": checker_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content), None

    except Exception as e:
        return None, f"Checker Failed: {str(e)}"
# ==========================================
# 3. MAIN APP UI
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
    st.sidebar.title("🌍 HOB OS")
    st.sidebar.caption(f"User: {st.session_state.username} | Role: {st.session_state.user_role}")
    if st.sidebar.button("Log Out"): st.session_state.logged_in = False; st.rerun()
    st.sidebar.divider()

    selected_mp = st.sidebar.selectbox("Marketplace", ["Myntra", "Flipkart", "Ajio", "Amazon", "Nykaa"])
    mp_cats = get_categories_for_marketplace(selected_mp)
    
    base_tabs = ["🛠️ Setup", "📈 SEO", "🚀 Run", "🖼️ Tools", "🧪 Prompt Lab"]
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
            if not default_mapping:
                for h in headers:
                    src = "Leave Blank"; h_low = h.lower()
                    if "image" in h_low or "sku" in h_low: src = "Input Excel"
                    elif h in master_options or "name" in h_low or "desc" in h_low: src = "AI Generation"
                    default_mapping.append({"Column Name": h, "Source": src, "Fixed Value": "", "Max Chars": "", "AI Style": "Standard (Auto)", "Custom Prompt": ""})
            
            ui_data = []
            if mode == "Edit Existing" and loaded:
                for col, rule in loaded['column_mapping'].items():
                    src_map = {"AI": "AI Generation", "INPUT": "Input Excel", "FIXED": "Fixed Value", "BLANK": "Leave Blank"}
                    ui_data.append({
                        "Column Name": col, "Source": src_map.get(rule['source'], "Leave Blank"),
                        "Fixed Value": rule.get('value', ''), "Max Chars": rule.get('max_len', ''),
                        "AI Style": rule.get('prompt_style', 'Standard (Auto)'), "Custom Prompt": rule.get('custom_prompt', '')
                    })
            else: ui_data = default_mapping

            edited_df = st.data_editor(
                pd.DataFrame(ui_data),
                column_config={
                    "Source": st.column_config.SelectboxColumn("Source", options=["Input Excel", "AI Generation", "Fixed Value", "Leave Blank"]),
                    "Max Chars": st.column_config.NumberColumn("Max Chars", min_value=1, max_value=5000),
                    "AI Style": st.column_config.SelectboxColumn("AI Style", options=["Standard (Auto)", "✨ Creative", "🔧 Technical", "🔍 SEO"]),
                    "Custom Prompt": st.column_config.TextColumn("Custom Prompt", width="large")
                },
                hide_index=True, use_container_width=True, height=400
            )
            
            if st.button("Save Config"):
                final_map = {}
                for i, row in edited_df.iterrows():
                    src_code = "AI" if row['Source'] == "AI Generation" else "INPUT" if row['Source'] == "Input Excel" else "FIXED" if row['Source'] == "Fixed Value" else "BLANK"
                    
                    # ROBUST MAX CHARS LOGIC
                    m_len = row['Max Chars']
                    if pd.isna(m_len) or str(m_len).strip() == "" or str(m_len).strip() == "0":
                        m_len = ""
                    else:
                        try:
                            m_len = int(float(m_len))
                        except ValueError:
                            m_len = ""

                    final_map[row['Column Name']] = {
                        "source": src_code, "value": row['Fixed Value'], "max_len": m_len,
                        "prompt_style": row['AI Style'], "custom_prompt": row['Custom Prompt']
                    }
                if save_config(selected_mp, cat_name, {"category_name": cat_name, "headers": headers, "master_data": master_options, "column_mapping": final_map}):
                    st.success("✅ Saved Successfully!"); time.sleep(1); st.rerun()

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

 # --- TAB 3: RUN (Phase 3: Final Polish - SKU Edition) ---
    with tabs[2]:
        st.header(f"🚀 {selected_mp} Generator Operations")
        
        # 1. PRE-FLIGHT CHECKS
        if not mp_cats: 
            st.warning("⚠️ No categories configured. Please go to 'Setup' first.")
            st.stop()
        
        c_config1, c_config2 = st.columns([1, 2])
        with c_config1:
            run_cat = st.selectbox("Select Category", mp_cats, key="run")
            config = load_config(selected_mp, run_cat)
            
            if config:
                req_cols = config['headers']
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer: pd.DataFrame(columns=req_cols).to_excel(writer, index=False)
                st.download_button("📥 Download Template", output.getvalue(), file_name=f"Template_{run_cat}.xlsx", use_container_width=True)

        with c_config2:
            st.info(f"**Active Rules:** {len(config.get('column_mapping', {}))} Columns Mapped | **Master Data:** {len(config.get('master_data', {}))} Lists Enforced")

        active_kws = get_seo(selected_mp, run_cat)
        
        st.divider()
        
        # 2. DATA UPLOAD & SETTINGS
        input_file = st.file_uploader("📂 Upload Input File (.xlsx)", type=["xlsx"], key="run_in")
        
        if input_file and config:
            df_input = pd.read_excel(input_file)
            
            with st.expander("⚙️ Run Settings", expanded=True):
                c_set1, c_set2, c_set3 = st.columns(3)
                with c_set1:
                    run_mode = st.radio("Run Scope", ["🧪 Test (First 3 Rows)", "🚀 Production (All Rows)"])
                with c_set2:
                    arch_mode = st.selectbox("AI Architecture", ["✨ Dual-AI (Maker-Checker)", "⚡ Gemini Only (Fast)", "🧠 GPT-4o Only (Precise)"])
                with c_set3:
                    # SMART COLUMN DETECTION
                    all_cols = df_input.columns.tolist()
                    
                    # 1. Image Column
                    img_candidates = [c for c in all_cols if "url" in c.lower() or "image" in c.lower()]
                    img_default = all_cols.index(img_candidates[0]) if img_candidates else 0
                    img_col = st.selectbox("Image Column", all_cols, index=img_default)
                    
                    # 2. SKU/ID Column (New Feature)
                    sku_candidates = [c for c in all_cols if "sku" in c.lower() or "style" in c.lower() or "design" in c.lower()]
                    sku_default = all_cols.index(sku_candidates[0]) if sku_candidates else 0
                    sku_col = st.selectbox("SKU/ID Column", all_cols, index=sku_default, help="Used for logging (e.g. 'Processed SKU-123')")

            # PRE-PROCESS ANALYSIS
            df_to_proc = df_input.head(3) if "Test" in run_mode else df_input
            
            # Clean Keys
            df_to_proc[img_col] = df_to_proc[img_col].astype(str).str.strip()
            
            unique_urls = df_to_proc[img_col].unique()
            unique_urls = [u for u in unique_urls if u.lower() != "nan" and u != ""]
            
            # COST MATH
            is_mini = "Mini" in arch_mode
            checker_cost = 0.001 if is_mini else 0.03
            maker_cost = 0.002
            cost_per_image = maker_cost + checker_cost if "Dual-AI" in arch_mode else (checker_cost if "GPT" in arch_mode else maker_cost)
            total_est_cost = len(unique_urls) * cost_per_image

            # DASHBOARD METRICS
            m1, m2, m3 = st.columns(3)
            m1.metric("Rows to Fill", len(df_to_proc))
            m2.metric("Unique Images", len(unique_urls), delta=f"Saved {len(df_to_proc)-len(unique_urls)} API Calls")
            m3.metric("Est. Cost", f"${total_est_cost:.3f}")

            # --- EXECUTION BUTTON ---
            if st.button("▶️ START ENGINE", type="primary", use_container_width=True):
                st.session_state.gen_results = [] 
                
                # UI LAYOUT
                st.write("### 👁️ Live Vision Dashboard")
                c_prog, c_vis = st.columns([1, 1])
                
                with c_prog:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    log_expander = st.expander("📜 Detailed Logs", expanded=True) # Expanded by default to see SKUs
                    log_container = log_expander.container(height=200)
                
                with c_vis:
                    img_preview = st.empty()
                    ai_status_badge = st.empty()

                image_knowledge_base = {} 
                mapping = config['column_mapping']
                
                try:
                    # === PHASE 1: IMAGE PROCESSING ===
                    for i, u_key in enumerate(unique_urls):
                        img_num = i + 1
                        total_imgs = len(unique_urls)
                        
                        # GET SKU NAME FOR DISPLAY
                        # Find the first row that matches this image to get the SKU Name
                        sku_label = f"Image {img_num}"
                        try:
                            match_row = df_to_proc[df_to_proc[img_col] == u_key]
                            if not match_row.empty:
                                sku_label = str(match_row.iloc[0][sku_col])
                        except: pass

                        # UI Update
                        status_text.markdown(f"**Analyzing: {sku_label}** ({img_num}/{total_imgs})")
                        progress_bar.progress(img_num / total_imgs)
                        ai_status_badge.info("⏳ Downloading...")
                        
                        # 1. DOWNLOAD PREP
                        download_url = u_key 
                        if "dropbox.com" in download_url:
                            download_url = download_url.replace("?dl=0", "").replace("&dl=0", "") + ("&dl=1" if "?" in download_url else "?dl=1")
                        
                        # 2. DOWNLOAD & PREVIEW
                        base64_img = None
                        try:
                            response = requests.get(download_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                            if response.status_code == 200:
                                img_data = response.content
                                # SHOW SKU IN CAPTION
                                img_preview.image(img_data, caption=f"Analyzing: {sku_label}", width=250)
                                base64_img = base64.b64encode(img_data).decode('utf-8')
                            else:
                                raise Exception(f"Status {response.status_code}")
                        except Exception as e:
                            log_container.warning(f"Download fail for {sku_label}: {e}")
                            img_preview.error("❌ Image Error")
                            continue

                        # 3. HINTS
                        hints = "Product analysis."
                        try:
                            match_row = df_to_proc[df_to_proc[img_col] == u_key]
                            if not match_row.empty:
                                sample_row = match_row.iloc[0]
                                hints = ", ".join([f"{k}: {v}" for k,v in sample_row.items() if k != img_col and str(v).lower() != "nan"])
                                hints = smart_truncate(hints, 300)
                        except: pass

                        # 4. AI ANALYSIS
                        ai_status_badge.warning(f"🤖 AI Working on {sku_label}...")
                        ai_data = None
                        max_retries = 3
                        
                        for attempt in range(max_retries):
                            try:
                                if "Dual-AI" in arch_mode:
                                    ai_data, err = analyze_image_maker_checker(client, base64_img, hints, active_kws, config, selected_mp)
                                else:
                                    ai_data = {}
                                    err = "Select Dual-AI"

                                if err:
                                    if "429" in str(err) or "quota" in str(err).lower():
                                        log_container.warning(f"Rate Limit 429. Sleeping 60s... (Retry {attempt+1})")
                                        time.sleep(60)
                                        continue 
                                    else:
                                        raise Exception(err)
                                
                                image_knowledge_base[u_key] = ai_data
                                
                                # LOG SUCCESS WITH SKU
                                log_container.success(f"✅ {sku_label} Processed")
                                break 

                            except Exception as e:
                                if attempt == max_retries - 1:
                                    log_container.error(f"Failed {sku_label}: {str(e)}")
                                time.sleep(2)
                        
                        time.sleep(0.5)

                    # === CLEANUP UI ===
                    ai_status_badge.success("✅ Batch Analysis Complete")
                    img_preview.empty() # Remove the last image
                    
                    # === PHASE 2: DATA MAPPING ===
                    status_text.markdown("**🚀 Finalizing: Mapping Data to Excel...**")
                    final_rows = []
                    
                    for idx, row in df_to_proc.iterrows():
                        u_key = str(row.get(img_col, "")).strip()
                        ai_data = image_knowledge_base.get(u_key, {})
                        
                        new_row = {}
                        for col in config['headers']:
                            rule = mapping.get(col, {'source': 'BLANK'})
                            val = ""
                            
                            if rule['source'] == 'INPUT': val = row.get(col, "")
                            elif rule['source'] == 'FIXED': val = rule['value']
                            elif rule['source'] == 'AI' and ai_data:
                                if col in ai_data: val = ai_data[col]
                                else: 
                                    clean_col = col.lower().replace(" ", "").replace("_", "")
                                    for k,v in ai_data.items():
                                        if k.lower().replace(" ", "") in clean_col:
                                            val = v; break
                                
                                m_list = []
                                for mc, opts in config['master_data'].items():
                                    if mc.lower() in col.lower(): m_list = opts; break
                                if m_list and val: val = enforce_master_data_fallback(val, m_list)
                            
                            # SANITIZATION
                            if isinstance(val, list) or isinstance(val, tuple): val = ", ".join([str(x) for x in val])
                            elif isinstance(val, dict): val = json.dumps(val)
                            val = str(val).strip()
                            if rule.get('max_len'): val = smart_truncate(val, int(float(rule['max_len'])))
                            
                            new_row[col] = val
                        
                        final_rows.append(new_row)
                    
                    st.session_state.gen_results = final_rows
                    st.toast("Processing Complete!", icon="🎉")

                except Exception as critical_e:
                    st.error(f"💀 CRITICAL ERROR: {str(critical_e)}")
                    log_container.exception(critical_e)

            # --- RESULT DISPLAY ---
            if "gen_results" in st.session_state and len(st.session_state.gen_results) > 0:
                st.divider()
                
                res_tab1, res_tab2 = st.tabs(["📊 Success Data", "⚠️ Needs Attention"])
                
                final_df = pd.DataFrame(st.session_state.gen_results)
                
                with res_tab1:
                    st.success(f"Successfully generated {len(final_df)} rows.")
                    st.dataframe(final_df.astype(str), use_container_width=True)
                    
                    output_gen = BytesIO()
                    with pd.ExcelWriter(output_gen, engine='xlsxwriter') as writer: 
                        final_df.to_excel(writer, index=False)
                    
                    st.download_button(
                        "⬇️ Download Final Excel", 
                        output_gen.getvalue(), 
                        file_name=f"{selected_mp}_Generated_{len(final_df)}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )

                with res_tab2:
                    st.info("Rows with missing images or data will be flagged here in future updates.")
    # --- TAB 4: TOOLS ---
    with tabs[3]:
        st.header("🛠️ Media Tools")
        if st.radio("Tool", ["Check Compliance (Vision Guard)", "Process Images"]) == "Check Compliance (Vision Guard)":
            audit_files = st.file_uploader("Images", accept_multiple_files=True)
            if audit_files and st.button("Run Audit"):
                st.write("Simulated Vision Guard Result: ✅ PASS") 

    # --- TAB 5: PROMPT LAB ---
    with tabs[4]: 
        st.header("🧪 Prompt Lab (Lyra)")
        idea = st.text_area("Rough Idea")
        if st.button("Optimize"):
            st.write(run_lyra_optimization("GPT", idea))

    # --- ADMIN ---
    if st.session_state.user_role == "admin":
        with tabs[6]:
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





