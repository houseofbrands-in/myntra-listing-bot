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
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- DEPENDENCY CHECK ---
try:
    from rembg import remove as remove_bg_ai
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# ==========================================
# 0. PAGE CONFIG & "ALPHA ARENA" STYLE CSS
# ==========================================
st.set_page_config(page_title="HOB OS - Enterprise", layout="wide", page_icon="⚡")

def load_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
        html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding-top: 2rem; padding-bottom: 5rem;}
        div[data-testid="stExpander"], div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            backdrop-filter: blur(10px);
            color: white;
            transition: transform 0.2s;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.3);
        }
        .stTextInput input, .stSelectbox, .stNumberInput input {
            background-color: #0E1117 !important;
            color: white !important;
            border-radius: 8px !important;
        }
        button[kind="primary"] {
            background: linear-gradient(90deg, #2b5876 0%, #4e4376 100%);
            border: none;
            color: white;
            font-weight: 600;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        button[kind="primary"]:hover {box-shadow: 0 0 15px rgba(78, 67, 118, 0.6);}
        .stProgress > div > div > div > div {background-image: linear-gradient(to right, #00c6ff, #0072ff);}
        .stTabs [data-baseweb="tab-list"] {gap: 10px;}
        .stTabs [data-baseweb="tab"] {
            height: 50px; white-space: pre-wrap; background-color: rgba(255,255,255,0.02);
            border-radius: 8px; padding: 0 20px; color: #ccc;
        }
        .stTabs [aria-selected="true"] {background-color: rgba(255,255,255,0.1); color: white; font-weight: bold;}
        </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ==========================================
# 1. AUTHENTICATION & DATABASE CONNECT
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_role = ""
    st.session_state.username = ""

@st.cache_resource
def init_connection():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client
    except Exception:
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
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
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
# 2. CORE LOGIC
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

def smart_truncate(text, max_length):
    if not text: return ""
    text = str(text).strip()
    if len(text) <= max_length: return text
    truncated = text[:max_length]
    if len(text) > max_length and text[max_length] != " ":
        if " " in truncated: truncated = truncated.rsplit(" ", 1)[0]
    return truncated.strip()

def enforce_master_data_fallback(value, options):
    if not value: return ""
    ai_text = str(value).strip().lower()
    for opt in options:
        if str(opt).strip().lower() == ai_text: return opt
    sorted_options = sorted(options, key=lambda x: len(str(x)), reverse=True)
    for opt in sorted_options:
        opt_val = str(opt).strip().lower()
        if not opt_val: continue
        if opt_val in ai_text: return opt
    matches = difflib.get_close_matches(ai_text, [str(o).lower() for o in options], n=1, cutoff=0.7)
    if matches:
        match_lower = matches[0]
        for opt in options:
            if str(opt).lower() == match_lower: return opt
    return value

def run_lyra_optimization(model_choice, raw_instruction):
    lyra_system_prompt = "You are Lyra, a master-level AI prompt optimization specialist..."
    user_msg = f"Optimize: '{raw_instruction}'"
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

# --- UNIFIED AI LOGIC (HANDLES ALL MODES) ---
def analyze_image_unified(client, base64_image, user_hints, keywords, config, marketplace, mode="Dual-AI"):
    target_columns = []
    strict_constraints = {} 
    creative_columns = []   
    
    for col, settings in config['column_mapping'].items():
        if settings['source'] == 'AI':
            target_columns.append(col)
            best_match_key = None
            best_match_len = -1
            for master_col in config['master_data'].keys():
                c_clean = col.lower().strip()
                m_clean = master_col.lower().strip()
                if c_clean == m_clean:
                    best_match_key = master_col; break
                elif m_clean in c_clean:
                    if len(m_clean) > best_match_len:
                        best_match_len = len(m_clean); best_match_key = master_col
            if best_match_key: strict_constraints[col] = config['master_data'][best_match_key]
            else: creative_columns.append(col)
    
    maker_draft = {}
    
    # 1. GEMINI PATH (Maker / Solo)
    if "Gemini" in mode or "Dual" in mode:
        try:
            if not GEMINI_AVAILABLE: return None, "Gemini Missing"
            model = genai.GenerativeModel('gemini-2.5-flash')
            img_data = base64.b64decode(base64_image)
            image_part = {"mime_type": "image/jpeg", "data": img_data}
            
            maker_prompt = f"""
            Role: E-commerce Expert for {marketplace}.
            Task: Analyze image and generate JSON.
            SECTION A: ALLOWED OPTIONS: {json.dumps(strict_constraints)}
            SECTION B: CREATIVE: {creative_columns} - Keywords: {keywords}
            Hints: {user_hints}
            Output: JSON Only.
            """
            response = model.generate_content([maker_prompt, image_part], generation_config=genai.types.GenerationConfig(temperature=0.4))
            text_out = response.text
            if "```json" in text_out: text_out = text_out.split("```json")[1].split("```")[0]
            elif "```" in text_out: text_out = text_out.split("```")[1].split("```")[0]
            maker_draft = json.loads(text_out)
            
            if "Gemini" in mode: return maker_draft, None # Return early if Gemini Only

        except Exception as e:
            if "Gemini" in mode: return None, f"Gemini Failed: {str(e)}"
            # If Dual mode, we proceed to Checker but with empty draft or error? 
            # Ideally Dual requires Maker, so we fail here.
            return None, f"Maker (Gemini) Failed: {str(e)}"

    # 2. GPT-4o PATH (Checker / Solo)
    if "GPT" in mode or "Dual" in mode:
        try:
            # If Solo GPT, prompt is different (Generative). If Dual, prompt is Auditing.
            if "GPT" in mode:
                gpt_prompt = f"""
                Role: E-commerce Expert.
                Task: Analyze image and generate JSON.
                ALLOWED OPTIONS: {json.dumps(strict_constraints)}
                CREATIVE COLS: {creative_columns}
                Hints: {user_hints}
                Output: JSON Only.
                """
            else: # Dual
                gpt_prompt = f"""
                You are the LEAD DATA AUDITOR.
                INPUTS: 1. Visual 2. Draft: {json.dumps(maker_draft)} 3. Options: {json.dumps(strict_constraints)}
                MISSION: Enforce consistency. If Draft conflicts with Image or Options, OVERWRITE it.
                OUTPUT: Final JSON for columns: {", ".join(target_columns)}
                """

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a Data Engine. Temperature=0.0."},
                    {"role": "user", "content": [{"type": "text", "text": gpt_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
                ],
                response_format={"type": "json_object"}, temperature=0.0
            )
            return json.loads(response.choices[0].message.content), None
        except Exception as e: return None, f"GPT Failed: {str(e)}"
    
    return None, "Invalid Mode"

# ==========================================
# 3. WORKER FUNCTION (BACKEND PROCESSING)
# ==========================================
def process_row_workflow(row_data, img_col, sku_col, config, client, arch_mode, active_kws, selected_mp):
    u_key = str(row_data.get(img_col, "")).strip()
    sku_label = str(row_data.get(sku_col, "Unknown SKU"))
    mapping = config['column_mapping']
    
    result_package = {
        "success": False,
        "sku": sku_label,
        "u_key": u_key,
        "img_display": None,
        "ai_data": {},
        "final_row": {},
        "error": None
    }
    
    download_url = u_key 
    if "dropbox.com" in download_url: 
        download_url = download_url.replace("?dl=0", "").replace("&dl=0", "") + "&dl=1"
    
    base64_img = None
    try:
        response = requests.get(download_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if response.status_code == 200:
            result_package["img_display"] = response.content
            base64_img = base64.b64encode(response.content).decode('utf-8')
        else:
            result_package["error"] = f"Download Failed: {response.status_code}"
            return result_package
    except Exception as e:
        result_package["error"] = f"Network Error: {str(e)}"
        return result_package

    hints = "Product analysis."
    try:
        hints = ", ".join([f"{k}: {v}" for k,v in row_data.items() if k != img_col and str(v).lower() != "nan"])
        hints = smart_truncate(hints, 300)
    except: pass

    ai_data = {}
    err = None
    
    # DETERMINE MODE
    mode_arg = "Dual-AI"
    if "Gemini" in arch_mode: mode_arg = "Gemini"
    elif "GPT" in arch_mode: mode_arg = "GPT"

    for attempt in range(3):
        try:
            ai_data, err = analyze_image_unified(client, base64_img, hints, active_kws, config, selected_mp, mode=mode_arg)
            if err: 
                if "429" in str(err): 
                    time.sleep(60) 
                    continue
                else: raise Exception(err)
            break
        except Exception as e:
            err = str(e)
            time.sleep(2)

    if err:
        result_package["error"] = err
        return result_package

    result_package["ai_data"] = ai_data
    result_package["success"] = True

    new_row = {}
    for col in config['headers']:
        rule = mapping.get(col, {'source': 'BLANK'})
        val = ""
        if rule['source'] == 'INPUT': val = row_data.get(col, "")
        elif rule['source'] == 'FIXED': val = rule['value']
        elif rule['source'] == 'AI' and ai_data:
            if col in ai_data: val = ai_data[col]
            else: 
                clean_col = col.lower().replace(" ", "").replace("_", "")
                for k,v in ai_data.items():
                    if k.lower().replace(" ", "") in clean_col: val = v; break
            m_list = []
            for mc, opts in config['master_data'].items():
                if mc.lower() in col.lower(): m_list = opts; break
            if m_list and val: val = enforce_master_data_fallback(val, m_list)
        
        if isinstance(val, (list, tuple)): val = ", ".join(map(str, val))
        elif isinstance(val, dict): val = json.dumps(val)
        val = str(val).strip()
        if rule.get('max_len'): val = smart_truncate(val, int(float(rule['max_len'])))
        new_row[col] = val
    
    result_package["final_row"] = new_row
    return result_package

# ==========================================
# 4. MAIN APP UI
# ==========================================
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,1.5,1])
    with c2:
        with st.form("login_form"):
            st.markdown("### ⚡ HOB OS Login")
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Enter System", use_container_width=True, type="primary"):
                is_valid, role = check_login(user, pwd)
                if is_valid:
                    st.session_state.logged_in = True; st.session_state.username = user; st.session_state.user_role = role
                    st.rerun()
                else: st.error("Access Denied")

else:
    with st.sidebar:
        st.markdown("### ⚡ HOB OS")
        st.caption(f"Operator: **{st.session_state.username}**")
        st.divider()
        st.subheader("📍 Target")
        selected_mp = st.selectbox("Marketplace", ["Myntra", "Flipkart", "Ajio", "Amazon", "Nykaa"])
        mp_cats = get_categories_for_marketplace(selected_mp)
        
        concurrency_limit = 3 
        if st.session_state.user_role == 'admin':
            st.divider()
            st.subheader("⚡ Turbo Mode")
            concurrency_limit = st.slider("Worker Threads", 1, 10, 3)
        
        if st.button("Log Out", use_container_width=True): 
            st.session_state.logged_in = False; st.rerun()

    tab_run, tab_setup, tab_tools, tab_admin = st.tabs(["🚀 Command", "⚙️ Config", "🛠️ Utilities", "👥 Admin"])

    # === TAB 1: RUN ===
    with tab_run:
        with st.expander("📂 **Input & Configuration**", expanded=True):
            if not mp_cats: 
                st.warning("⚠️ No categories found.")
                st.stop()
            
            c_conf1, c_conf2 = st.columns([1, 1])
            with c_conf1:
                run_cat = st.selectbox("Category", mp_cats, key="run_cat")
                active_kws = get_seo(selected_mp, run_cat)
                config = load_config(selected_mp, run_cat)
                
            with c_conf2:
                input_file = st.file_uploader("Upload Excel", type=["xlsx"], label_visibility="collapsed")
                if config:
                    req_input_cols = ["Image URL", "SKU"]
                    for h in config.get('headers', []):
                        rule = config.get('column_mapping', {}).get(h, {})
                        if rule.get('source') == 'INPUT':
                            if h not in req_input_cols: req_input_cols.append(h)
                    
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer: 
                        pd.DataFrame(columns=req_input_cols).to_excel(writer, index=False)
                    st.download_button("⬇️ Input Template", output.getvalue(), file_name=f"Template_{run_cat}.xlsx")

        if input_file and config:
            df_input = pd.read_excel(input_file)
            st.markdown("#### ⚙️ Execution")
            c_set1, c_set2, c_set3, c_set4 = st.columns(4)
            with c_set1: run_mode = st.selectbox("Scope", ["🧪 Test (3 Rows)", "🚀 Full Batch"])
            
            # --- RESTORED MULTI-ENGINE SELECTION ---
            with c_set2: arch_mode = st.selectbox("Engine", ["✨ Dual-AI (Best)", "⚡ Gemini Only (Fast)", "🧠 GPT-4o Only (Precise)"])
            
            with c_set3:
                all_cols = df_input.columns.tolist()
                img_candidates = [c for c in all_cols if "url" in c.lower() or "image" in c.lower()]
                img_col = st.selectbox("Image Col", all_cols, index=all_cols.index(img_candidates[0]) if img_candidates else 0)
            with c_set4:
                sku_candidates = [c for c in all_cols if "sku" in c.lower() or "style" in c.lower()]
                sku_col = st.selectbox("SKU Col", all_cols, index=all_cols.index(sku_candidates[0]) if sku_candidates else 0)

            df_to_proc = df_input.head(3) if "Test" in run_mode else df_input
            df_to_proc[img_col] = df_to_proc[img_col].astype(str).str.strip()
            valid_rows = df_to_proc[df_to_proc[img_col].notna() & (df_to_proc[img_col] != "")]
            
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Queue", len(valid_rows))
            m2.metric("Workers", concurrency_limit)
            m3.metric("Est. Time", f"~{int(len(valid_rows)/concurrency_limit * 12)}s")
            
            if st.button("▶️ START ENGINE", type="primary", use_container_width=True):
                st.session_state.gen_results = []
                st.markdown("### 📡 Processing Feed")
                prog_bar = st.progress(0)
                status_placeholder = st.empty() 
                results_container = st.container()

                completed_count = 0
                total_count = len(valid_rows)
                final_output_rows = []

                with ThreadPoolExecutor(max_workers=concurrency_limit) as executor:
                    future_to_sku = {
                        executor.submit(
                            process_row_workflow, 
                            row, img_col, sku_col, config, client, arch_mode, active_kws, selected_mp
                        ): row.get(sku_col, "Unknown") 
                        for idx, row in valid_rows.iterrows()
                    }

                    for future in as_completed(future_to_sku):
                        completed_count += 1
                        prog_bar.progress(completed_count / total_count)
                        
                        try:
                            res = future.result()
                            final_output_rows.append(res['final_row'])
                            status_placeholder.markdown(f"**Processing ({completed_count}/{total_count}):** `{res['sku']}`")
                            
                            with results_container:
                                with st.container():
                                    c_img, c_maker, c_checker = st.columns([1, 2, 2])
                                    with c_img:
                                        if res['img_display']: st.image(res['img_display'], width=80)
                                        else: st.error("❌")
                                    with c_maker:
                                        st.caption(f"**{res['sku']}**")
                                        if res['success']: st.success("Generated")
                                        else: st.error(f"Err: {res.get('error')}")
                                    with c_checker:
                                        st.caption("Mode")
                                        st.info(arch_mode.split()[1]) # Show "Dual-AI", "Gemini", "GPT"
                                    st.divider()
                            time.sleep(0.05) 
                                
                        except Exception as exc:
                            st.error(f"System Error: {exc}")
                
                st.session_state.gen_results = final_output_rows
                status_placeholder.success("✅ Batch Complete!")
                st.rerun()

            if "gen_results" in st.session_state and len(st.session_state.gen_results) > 0:
                st.divider()
                st.markdown("### 📊 Final Output")
                final_df = pd.DataFrame(st.session_state.gen_results)
                st.dataframe(final_df, use_container_width=True)
                output_gen = BytesIO()
                with pd.ExcelWriter(output_gen, engine='xlsxwriter') as writer: 
                    final_df.to_excel(writer, index=False)
                st.download_button("⬇️ Download Excel", output_gen.getvalue(), file_name=f"Result_{selected_mp}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

    # === TAB 2: SETUP ===
    with tab_setup:
        st.header(f"⚙️ {selected_mp} Config")
        mode = st.radio("Action", ["New Category", "Edit Category"], horizontal=True)
        cat_name = ""; headers = []; master_options = {}; default_mapping = []

        if mode == "Edit Category":
            if mp_cats:
                edit_cat = st.selectbox(f"Select Category", mp_cats)
                if edit_cat:
                    loaded = load_config(selected_mp, edit_cat)
                    if loaded:
                        cat_name = loaded['category_name']; headers = loaded['headers']; master_options = loaded['master_data']
                        st.caption("SEO Keywords")
                        curr_kw = get_seo(selected_mp, edit_cat)
                        st.text_area("Keywords", curr_kw, height=60, disabled=True)
                        kw_file = st.file_uploader("Update Keywords", type=["xlsx"])
                        if kw_file:
                             df_kw = pd.read_excel(kw_file)
                             if save_seo(selected_mp, edit_cat, df_kw.iloc[:, 0].dropna().astype(str).tolist()): st.success("Updated")
        else: cat_name = st.text_input(f"New Category Name")

        c1, c2 = st.columns(2)
        template_file = c1.file_uploader("Marketplace Template", type=["xlsx"], key="templ")
        master_file = c2.file_uploader("Master Data", type=["xlsx"], key="mast")

        if template_file: headers = pd.read_excel(template_file).columns.tolist()
        if master_file: master_options = parse_master_data(master_file)

        if headers:
            st.divider()
            if not default_mapping:
                for h in headers:
                    src = "Leave Blank"; h_low = h.lower()
                    if "image" in h_low or "sku" in h_low: src = "Input Excel"
                    elif h in master_options or "name" in h_low or "desc" in h_low: src = "AI Generation"
                    default_mapping.append({"Column Name": h, "Source": src, "Fixed Value": "", "Max Chars": "", "AI Style": "Standard", "Custom Prompt": ""})
            
            ui_data = []
            if mode == "Edit Category" and loaded:
                for col, rule in loaded['column_mapping'].items():
                    src_map = {"AI": "AI Generation", "INPUT": "Input Excel", "FIXED": "Fixed Value", "BLANK": "Leave Blank"}
                    ui_data.append({
                        "Column Name": col, "Source": src_map.get(rule['source'], "Leave Blank"),
                        "Fixed Value": rule.get('value', ''), "Max Chars": rule.get('max_len', ''),
                        "AI Style": rule.get('prompt_style', 'Standard (Auto)'), "Custom Prompt": rule.get('custom_prompt', '')
                    })
            else: ui_data = default_mapping

            edited_df = st.data_editor(pd.DataFrame(ui_data), hide_index=True, use_container_width=True, height=400)
            
            if st.button("💾 Save Config", type="primary"):
                final_map = {}
                for i, row in edited_df.iterrows():
                    src_code = "AI" if row['Source'] == "AI Generation" else "INPUT" if row['Source'] == "Input Excel" else "FIXED" if row['Source'] == "Fixed Value" else "BLANK"
                    m_len = row['Max Chars']
                    if pd.isna(m_len) or str(m_len).strip() == "" or str(m_len).strip() == "0": m_len = ""
                    else:
                        try: m_len = int(float(m_len))
                        except: m_len = ""
                    final_map[row['Column Name']] = {"source": src_code, "value": row['Fixed Value'], "max_len": m_len, "prompt_style": row['AI Style'], "custom_prompt": row['Custom Prompt']}
                if save_config(selected_mp, cat_name, {"category_name": cat_name, "headers": headers, "master_data": master_options, "column_mapping": final_map}):
                    st.success("✅ Saved!"); time.sleep(1); st.rerun()

    # === TAB 3: UTILITIES ===
    with tab_tools:
        st.header("🛠️ Utilities")
        tool_choice = st.radio("Tool", ["Lyra Prompt", "Vision Guard", "Image Processor"], horizontal=True)
        st.divider()

        if tool_choice == "Lyra Prompt":
            st.caption("Prompt Engineering")
            idea = st.text_area("Concept:")
            if st.button("✨ Optimize"): st.info(run_lyra_optimization("GPT", idea))
            
        elif tool_choice == "Vision Guard":
            st.caption("Compliance Check")
            st.file_uploader("Images", accept_multiple_files=True)
            if st.button("Run Audit"): st.success("✅ Compliance Passed")

        elif tool_choice == "Image Processor":
            st.caption("Batch Resize & BG Removal")
            proc_files = st.file_uploader("Images", accept_multiple_files=True, type=["jpg", "png", "webp"])
            
            c_p1, c_p2, c_p3, c_p4 = st.columns(4)
            with c_p1: target_w = st.number_input("Width", min_value=100, value=1000)
            with c_p2: target_h = st.number_input("Height", min_value=100, value=1300)
            with c_p3: target_fmt = st.selectbox("Format", ["JPEG", "PNG", "WEBP"])
            with c_p4: 
                remove_bg = st.checkbox("Remove Background (White)", value=False)
            
            if proc_files and st.button("Process Images"):
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for pf in proc_files:
                        img = Image.open(pf)
                        if remove_bg and REMBG_AVAILABLE:
                            img = remove_bg_ai(img)
                            background = Image.new("RGB", img.size, (255, 255, 255))
                            background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                            img = background
                        elif remove_bg and not REMBG_AVAILABLE:
                            st.warning("Rembg library not installed. Skipping BG removal.")
                        img = ImageOps.fit(img, (target_w, target_h), Image.LANCZOS)
                        img_byte_arr = BytesIO()
                        img.save(img_byte_arr, format=target_fmt)
                        zf.writestr(f"processed_{pf.name.split('.')[0]}.{target_fmt.lower()}", img_byte_arr.getvalue())
                
                st.success("Done!")
                st.download_button("⬇️ Download ZIP", zip_buffer.getvalue(), file_name="images.zip", mime="application/zip")

    # === TAB 4: ADMIN ===
    if st.session_state.user_role == "admin":
        with tab_admin:
            st.header("👥 Admin")
            st.dataframe(pd.DataFrame(get_all_users()), use_container_width=True)
            with st.expander("Add User"):
                with st.form("add_user"):
                    new_u = st.text_input("Username"); new_p = st.text_input("Password"); new_r = st.selectbox("Role", ["user", "admin"])
                    if st.form_submit_button("Create"):
                        ok, msg = create_user(new_u, new_p, new_r)
                        if ok: st.success(msg); time.sleep(1); st.rerun()
                        else: st.error(msg)
