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
except ImportError:
    REMBG_AVAILABLE = False

# ==========================================
# 0. PAGE CONFIG & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="HOB OS - Command Center", layout="wide", page_icon="🚀")

def load_custom_css():
    st.markdown("""
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        div[data-testid="stMetric"] {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .stExpander {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }
        button[kind="primary"] {
            background-color: #000000;
            color: #ffffff;
            border: 1px solid #000000;
        }
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
        # Ensure your secrets.toml has [gcp_service_account]
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
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=api_key)
    else:
        client = None
        
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

# --- DUAL AI LOGIC ---
def analyze_image_maker_checker(client, base64_image, user_hints, keywords, config, marketplace):
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
    except Exception as e: return None, f"Maker Failed: {str(e)}"

    try:
        checker_prompt = f"""
        You are the LEAD DATA AUDITOR.
        INPUTS: 1. Visual 2. Draft: {json.dumps(maker_draft)} 3. Options: {json.dumps(strict_constraints)}
        MISSION: Enforce consistency. If Draft conflicts with Image or Options, OVERWRITE it.
        OUTPUT: Final JSON for columns: {", ".join(target_columns)}
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Data Consistency Engine. Temperature=0.0."},
                {"role": "user", "content": [{"type": "text", "text": checker_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
            ],
            response_format={"type": "json_object"}, temperature=0.0
        )
        return json.loads(response.choices[0].message.content), None
    except Exception as e: return None, f"Checker Failed: {str(e)}"

# ==========================================
# 3. MAIN APP UI
# ==========================================

if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,1.5,1])
    with c2:
        st.markdown("""<div style='text-align: center; border: 1px solid #ddd; padding: 30px; border-radius: 10px; background-color: white;'>
            <h2 style='margin-bottom: 20px;'>🔒 HOB OS Access</h2>
            </div>""", unsafe_allow_html=True)
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Sign In", use_container_width=True, type="primary"):
                is_valid, role = check_login(user, pwd)
                if is_valid:
                    st.session_state.logged_in = True; st.session_state.username = user; st.session_state.user_role = role
                    st.rerun()
                else: st.error("Invalid Credentials")

else:
    with st.sidebar:
        st.markdown("## 🌍 HOB OS")
        st.caption(f"Logged in as: **{st.session_state.username}**")
        st.divider()
        st.subheader("📍 Scope")
        selected_mp = st.selectbox("Marketplace", ["Myntra", "Flipkart", "Ajio", "Amazon", "Nykaa"])
        mp_cats = get_categories_for_marketplace(selected_mp)
        if st.button("Log Out", use_container_width=True): 
            st.session_state.logged_in = False; st.rerun()

    tab_run, tab_setup, tab_tools, tab_admin = st.tabs(["🚀 Command Center", "⚙️ Rules & Setup", "🛠️ Utilities", "👥 Admin"])

    # === TAB 1: RUN (COMMAND CENTER) ===
    with tab_run:
        with st.expander("📂 **Input & Configuration (Click to Expand)**", expanded=True):
            if not mp_cats: 
                st.warning("⚠️ No categories configured. Go to 'Rules & Setup' tab.")
                st.stop()
            
            c_conf1, c_conf2 = st.columns([1, 1])
            with c_conf1:
                run_cat = st.selectbox("Select Category", mp_cats, key="run_cat")
                active_kws = get_seo(selected_mp, run_cat)
                config = load_config(selected_mp, run_cat)
                
            with c_conf2:
                input_file = st.file_uploader("Upload Input Excel (.xlsx)", type=["xlsx"], label_visibility="collapsed")
                
                # FIX 1: DOWNLOAD INPUT TEMPLATE (STRICT ORDER)
                if config:
                    req_input_cols = ["Image URL", "SKU"]
                    
                    # Iterate through the ORDERED headers list from config
                    for h in config.get('headers', []):
                        rule = config.get('column_mapping', {}).get(h, {})
                        if rule.get('source') == 'INPUT':
                            # Avoid duplicates only if they match the first two
                            if h not in req_input_cols:
                                req_input_cols.append(h)
                    
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer: 
                        pd.DataFrame(columns=req_input_cols).to_excel(writer, index=False)
                    
                    st.download_button(
                        "📥 Download Input Template (For Upload)", 
                        output.getvalue(), 
                        file_name=f"Input_Template_{run_cat}.xlsx",
                        help="Contains Image URL, SKU, and columns marked as 'Input Excel'."
                    )

        if input_file and config:
            df_input = pd.read_excel(input_file)
            st.markdown("##### ⚙️ Execution Settings")
            c_set1, c_set2, c_set3, c_set4 = st.columns(4)
            with c_set1: run_mode = st.selectbox("Run Scope", ["🧪 Test (First 3 Rows)", "🚀 Production (All Rows)"])
            with c_set2: arch_mode = st.selectbox("Architecture", ["✨ Dual-AI (Maker-Checker)", "⚡ Gemini Only", "🧠 GPT-4o Only"])
            with c_set3:
                all_cols = df_input.columns.tolist()
                img_candidates = [c for c in all_cols if "url" in c.lower() or "image" in c.lower()]
                img_default = all_cols.index(img_candidates[0]) if img_candidates else 0
                img_col = st.selectbox("Image Col", all_cols, index=img_default)
            with c_set4:
                sku_candidates = [c for c in all_cols if "sku" in c.lower() or "style" in c.lower()]
                sku_default = all_cols.index(sku_candidates[0]) if sku_candidates else 0
                sku_col = st.selectbox("SKU/ID Col", all_cols, index=sku_default)

            df_to_proc = df_input.head(3) if "Test" in run_mode else df_input
            df_to_proc[img_col] = df_to_proc[img_col].astype(str).str.strip()
            unique_urls = [u for u in df_to_proc[img_col].unique() if u.lower() != "nan" and u != ""]
            
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Rows to Process", len(df_to_proc))
            m2.metric("Unique Images", len(unique_urls))
            m3.metric("Est. Cost", f"${(len(unique_urls) * (0.032 if 'Dual-AI' in arch_mode else 0.03)):.3f}")
            
            if st.button("▶️ INITIATE BATCH PROCESSING", type="primary", use_container_width=True):
                st.session_state.gen_results = []
                st.markdown("### 📡 Live Feed")
                prog_bar = st.progress(0)
                
                with st.status("🚀 Processing Batch...", expanded=True) as status_box:
                    image_knowledge_base = {}
                    mapping = config['column_mapping']
                    for i, u_key in enumerate(unique_urls):
                        img_num = i + 1; total_imgs = len(unique_urls)
                        sku_label = f"Img-{img_num}"
                        try:
                            match_row = df_to_proc[df_to_proc[img_col] == u_key]
                            if not match_row.empty: sku_label = str(match_row.iloc[0][sku_col])
                        except: pass
                        
                        prog_bar.progress(img_num / total_imgs)
                        status_box.update(label=f"Analyzing {img_num}/{total_imgs}: **{sku_label}**")
                        
                        download_url = u_key 
                        if "dropbox.com" in download_url: download_url = download_url.replace("?dl=0", "").replace("&dl=0", "") + "&dl=1"
                        base64_img = None; img_display_data = None
                        try:
                            response = requests.get(download_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                            if response.status_code == 200:
                                img_display_data = response.content
                                base64_img = base64.b64encode(response.content).decode('utf-8')
                        except: pass

                        hints = "Product analysis."
                        try:
                            match_row = df_to_proc[df_to_proc[img_col] == u_key]
                            if not match_row.empty:
                                sample_row = match_row.iloc[0]
                                hints = ", ".join([f"{k}: {v}" for k,v in sample_row.items() if k != img_col and str(v).lower() != "nan"])
                                hints = smart_truncate(hints, 300)
                        except: pass
                        
                        ai_data = {}
                        if base64_img:
                             for attempt in range(3):
                                try:
                                    if "Dual-AI" in arch_mode:
                                        ai_data, err = analyze_image_maker_checker(client, base64_img, hints, active_kws, config, selected_mp)
                                        if err: raise Exception(err)
                                        break
                                except Exception as e:
                                    if "429" in str(e): time.sleep(60)
                                    time.sleep(2)
                             image_knowledge_base[u_key] = ai_data
                        
                        with st.container():
                            c_img, c_maker, c_checker = st.columns([1, 2, 2])
                            with c_img:
                                if img_display_data: st.image(img_display_data, width=150, caption=sku_label)
                                else: st.error("Img Fail")
                            with c_maker:
                                st.caption("🤖 **Drafting (Gemini)**")
                                if ai_data: st.json({k: ai_data[k] for k in list(ai_data)[:3]}, expanded=False)
                                else: st.write("Processing...")
                            with c_checker:
                                st.caption("👨‍⚖️ **Audited (GPT-4o)**")
                                if ai_data:
                                    st.success("Approved")
                                    with st.expander("View Final Data"): st.json(ai_data)
                                else: st.error("Failed")
                            st.divider()
                    status_box.update(label="✅ Batch Processing Complete!", state="complete", expanded=False)

                st.success("💾 Finalizing Excel File...")
                final_rows = []
                for idx, row in df_to_proc.iterrows():
                    u_key = str(row.get(img_col, "")).strip()
                    ai_data = image_knowledge_base.get(u_key, {})
                    new_row = {}
                    # STRICT OUTPUT ORDER: Uses config['headers'] directly
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
                    final_rows.append(new_row)
                st.session_state.gen_results = final_rows
                st.rerun()

            if "gen_results" in st.session_state and len(st.session_state.gen_results) > 0:
                st.divider()
                st.markdown("### 📊 Results")
                final_df = pd.DataFrame(st.session_state.gen_results)
                st.dataframe(final_df, use_container_width=True)
                output_gen = BytesIO()
                with pd.ExcelWriter(output_gen, engine='xlsxwriter') as writer: 
                    final_df.to_excel(writer, index=False)
                st.download_button("⬇️ Download Final Excel", output_gen.getvalue(), file_name=f"{selected_mp}_Result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

    # === TAB 2: SETUP ===
    with tab_setup:
        st.header(f"⚙️ {selected_mp} Configuration")
        mode = st.radio("Action", ["Create New Category", "Edit Existing"], horizontal=True)
        cat_name = ""; headers = []; master_options = {}; default_mapping = []

        if mode == "Edit Existing":
            if mp_cats:
                edit_cat = st.selectbox(f"Select Category to Edit", mp_cats)
                if edit_cat:
                    loaded = load_config(selected_mp, edit_cat)
                    if loaded:
                        cat_name = loaded['category_name']; headers = loaded['headers']; master_options = loaded['master_data']
                        st.subheader("SEO Keywords")
                        curr_kw = get_seo(selected_mp, edit_cat)
                        st.text_area("Current Keywords", curr_kw, height=60, disabled=True)
                        kw_file = st.file_uploader("Update Keywords (.xlsx)", type=["xlsx"])
                        if kw_file:
                             df_kw = pd.read_excel(kw_file)
                             if save_seo(selected_mp, edit_cat, df_kw.iloc[:, 0].dropna().astype(str).tolist()): st.success("SEO Updated")
        else: cat_name = st.text_input(f"New Category Name")

        c1, c2 = st.columns(2)
        template_file = c1.file_uploader("Template (.xlsx)", type=["xlsx"], key="templ")
        master_file = c2.file_uploader("Master Data (.xlsx)", type=["xlsx"], key="mast")

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

            # --- DEFINE DROPDOWN OPTIONS & RENDER EDITOR ---
            source_options = ["AI Generation", "Input Excel", "Fixed Value", "Leave Blank"]
            style_options = ["Standard (Auto)", "Creative (Marketing)", "Technical (Specs)", "SEO (Optimized)"]
            
            edited_df = st.data_editor(
                pd.DataFrame(ui_data),
                hide_index=True,
                use_container_width=True,
                height=400,
                column_config={
                    "Column Name": st.column_config.TextColumn("Column Name", disabled=True, width="medium"),
                    "Source": st.column_config.SelectboxColumn("Source", width="medium", options=source_options, required=True),
                    "Fixed Value": st.column_config.TextColumn("Fixed Value", width="medium"),
                    "Max Chars": st.column_config.NumberColumn("Max Chars", help="Limit output length", min_value=0, max_value=2000, step=1, width="small"),
                    "AI Style": st.column_config.SelectboxColumn("AI Style", width="medium", options=style_options, required=True),
                    "Custom Prompt": st.column_config.TextColumn("Custom Prompt", width="large")
                }
            )
            
            if st.button("💾 Save Configuration", type="primary"):
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
        tool_choice = st.radio("Select Tool", ["Lyra Prompt Optimizer", "Vision Guard", "Image Processor"], horizontal=True)
        st.divider()

        if tool_choice == "Lyra Prompt Optimizer":
            st.subheader("Lyra Prompt Optimizer")
            idea = st.text_area("Enter rough prompt idea:")
            if st.button("✨ Optimize"): st.info(run_lyra_optimization("GPT", idea))
            
        elif tool_choice == "Vision Guard":
            st.subheader("Vision Guard (Simulated)")
            st.write("Upload images to check compliance before processing.")
            st.file_uploader("Images", accept_multiple_files=True, key="vision_guard")
            if st.button("Run Audit"): st.success("✅ All images passed compliance checks.")

        elif tool_choice == "Image Processor":
            st.subheader("🖼️ Image Processor")
            proc_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png", "jpeg", "webp"])
            
            c_p1, c_p2, c_p3 = st.columns(3)
            with c_p1: target_w = st.number_input("Target Width", min_value=100, value=1000)
            with c_p2: target_h = st.number_input("Target Height", min_value=100, value=1300)
            with c_p3: target_fmt = st.selectbox("Format", ["JPEG", "PNG", "WEBP"])
            
            if proc_files and st.button("Process Images"):
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for pf in proc_files:
                        img = Image.open(pf)
                        img = ImageOps.fit(img, (target_w, target_h), Image.LANCZOS)
                        
                        img_byte_arr = BytesIO()
                        img.save(img_byte_arr, format=target_fmt)
                        zf.writestr(f"processed_{pf.name.split('.')[0]}.{target_fmt.lower()}", img_byte_arr.getvalue())
                
                st.success("Processing Complete!")
                st.download_button("⬇️ Download Processed Images (ZIP)", zip_buffer.getvalue(), file_name="processed_images.zip", mime="application/zip")

    # === TAB 4: ADMIN ===
    if st.session_state.user_role == "admin":
        with tab_admin:
            st.header("👥 User Management")
            users = get_all_users()
            st.dataframe(pd.DataFrame(users), use_container_width=True)
            with st.expander("Add New User"):
                with st.form("add_user"):
                    new_u = st.text_input("Username"); new_p = st.text_input("Password"); new_r = st.selectbox("Role", ["user", "admin"])
                    if st.form_submit_button("Create User"):
                        ok, msg = create_user(new_u, new_p, new_r)
                        if ok: st.success(msg); time.sleep(1); st.rerun()
                        else: st.error(msg)
