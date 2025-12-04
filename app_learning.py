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

st.set_page_config(page_title="HOB OS - V11 (Learning Beta)", layout="wide", page_icon="🧠")

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

def get_worksheet_object(ws_name):
    gc = init_connection()
    return gc.open("Agency_OS_Database").worksheet(ws_name)

SHEET_NAME = "Agency_OS_Database"

try:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except Exception as e:
    st.error(f"❌ Secrets Error: {str(e)}")
    st.stop()

# ==========================================
# 2. LEARNING MODULE (LML Logic)
# ==========================================

def save_visual_example(marketplace, category, attribute, correct_val, img_url, desc):
    """Saves a 'Truth' example to the Visual Glossary."""
    try:
        ws = get_worksheet_object("Visual_Glossary")
        ws.append_row([marketplace, category, attribute, correct_val, img_url, desc])
        st.cache_data.clear() # Clear cache so the AI can find it immediately
        return True
    except Exception as e:
        st.error(f"Save Error: {e}")
        return False

def get_visual_context(marketplace, category, attribute):
    """Fetches reference examples for a specific attribute."""
    rows = get_worksheet_data(SHEET_NAME, "Visual_Glossary")
    examples = []
    for r in rows[1:]: # Skip header
        # Check match (r[0]=MP, r[1]=Cat, r[2]=Attr)
        if r[0] == marketplace and r[1] == category and r[2].lower() == attribute.lower():
            examples.append({
                "value": r[3],
                "url": r[4],
                "desc": r[5]
            })
    return examples

# ==========================================
# 3. UTILS & AI CORE
# ==========================================

def encode_image_from_url(url):
    try:
        if pd.isna(url) or str(url).strip() == "": return None, "Empty URL"
        url = str(url).strip()
        if "dropbox.com" in url: url = url.replace("?dl=0", "").replace("&dl=0", "") + "&dl=1"
        if "drive.google.com" in url and "/view" in url:
            file_id = url.split("/d/")[1].split("/")[0]
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.google.com/"}
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8'), None
        return None, f"Status {response.status_code}"
    except Exception as e: return None, str(e)

# --- MASTER DATA ENFORCER (THE MISSING FUNCTION) ---
def enforce_master_data(value, options):
    """Forces AI value to closest match in Master Data list."""
    if not value: return ""
    str_val = str(value).strip()
    val_lower = str_val.lower()
    
    # 1. Exact Match
    for opt in options:
        if str(opt).lower() == val_lower:
            return opt 

    # 2. Synonyms (Fashion Logic)
    synonyms = {
        "no sleeve": "Sleeveless",
        "without sleeve": "Sleeveless",
        "half sleeve": "Short Sleeves",
        "full sleeve": "Long Sleeves",
        "print": "Printed",
        "solid color": "Solid",
        "plain": "Solid",
        "button down": "Button",
        "round": "Round Neck",
        "v-neck": "V Neck",
        "collar": "Collared"
    }
    for wrong, right in synonyms.items():
        if wrong in val_lower:
            for opt in options:
                if right.lower() in str(opt).lower(): return opt

    # 3. Partial
    for opt in options:
        if str(opt).lower() in val_lower: return opt

    # 4. Fuzzy
    matches = difflib.get_close_matches(str_val, [str(o) for o in options], n=1, cutoff=0.4)
    if matches: return matches[0]
        
    return ""

def analyze_image_with_learning(model_choice, client, image_url, user_hints, keywords, config, marketplace):
    # 1. Prepare Main Image
    base64_main, error = encode_image_from_url(image_url)
    if error: return None, error

    # 2. Build Schema & Fetch Visual References
    tech_cols = []
    creative_cols = []
    gemini_constraints = []
    relevant_options = {}
    
    # --- LEARNING INJECTION ---
    visual_references_prompt = ""
    
    for col, settings in config['column_mapping'].items():
        if settings['source'] == 'AI':
            master_opts = []
            for m_col, opts in config['master_data'].items():
                if m_col.lower() in col.lower() or col.lower() in m_col.lower():
                    master_opts = opts
                    
                    # CHECk GLOSSARY FOR THIS ATTRIBUTE
                    examples = get_visual_context(marketplace, config.get('category_name', ''), col)
                    if examples:
                        visual_references_prompt += f"\n\n--- REFERENCE EXAMPLES FOR '{col}' ---\n"
                        for ex in examples[:3]: # Limit to 3 examples
                            visual_references_prompt += f"- When image looks like [ {ex['desc']} ], the value is '{ex['value']}'.\n"
                    break
            
            if master_opts:
                tech_cols.append(col)
                relevant_options[col] = master_opts
                gemini_constraints.append(f"- Column '{col}': MUST be one of {json.dumps(master_opts)}")
            else:
                creative_cols.append(col)

    all_targets = tech_cols + creative_cols
    seo_section = f"SEO KEYWORDS: {keywords}" if keywords else ""
    
    # 3. The Prompt (Enhanced with Learning)
    prompt = f"""
    You are a Learning AI for {marketplace}.
    TASK: Generate JSON for: {all_targets}
    
    CONTEXT: {user_hints}
    {seo_section}
    
    *** VISUAL LEARNING MEMORY ***
    {visual_references_prompt}
    
    STRICT DATA RULES:
    {json.dumps(relevant_options)}
    
    OUTPUT: JSON Only.
    """

    try:
        if "GPT" in model_choice:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a JSON-only assistant."},
                    {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_main}"}}]}
                ],
                response_format={"type": "json_object"},
                max_tokens=1500
            )
            return json.loads(response.choices[0].message.content), None

        elif "Gemini" in model_choice:
            if not GEMINI_AVAILABLE: return None, "Gemini Key Missing"
            model = genai.GenerativeModel('gemini-2.5-flash') 
            img_data = base64.b64decode(base64_main)
            image_part = {"mime_type": "image/jpeg", "data": img_data}
            
            gemini_prompt = f"""
            {prompt}
            
            STRICT RULES:
            {chr(10).join(gemini_constraints)}
            
            Return JSON ONLY.
            """
            response = model.generate_content([gemini_prompt, image_part])
            text_out = response.text.replace("```json", "").replace("```", "")
            return json.loads(text_out), None

    except Exception as e: return None, str(e)
    return None, "Unknown Error"

# --- HELPER FUNCTIONS ---
def load_config(marketplace, category):
    rows = get_worksheet_data(SHEET_NAME, "Configs")
    for row in rows:
        if len(row) > 2 and row[0] == marketplace and row[1] == category:
            return json.loads(row[2])
    return None

def get_categories(marketplace):
    rows = get_worksheet_data(SHEET_NAME, "Configs")
    cats = [row[1] for row in rows if len(row) > 1 and row[0] == marketplace]
    return list(set([c for c in cats if c != "Category"]))

def check_login(u, p):
    rows = get_worksheet_data(SHEET_NAME, "Users")
    if not rows: return False, None
    for row in rows[1:]:
        if str(row[0]).strip() == u and str(row[1]).strip() == p: return True, row[2]
    return False, None

def get_seo(marketplace, category):
    rows = get_worksheet_data(SHEET_NAME, "SEO_Data")
    for row in rows:
        if len(row) > 2 and row[0] == marketplace and row[1] == category:
            return row[2]
    return ""

# ==========================================
# 4. APP INTERFACE
# ==========================================

if not st.session_state.logged_in:
    st.title("🔒 HOB OS (Learning Beta)")
    u = st.text_input("User"); p = st.text_input("Pass", type="password")
    if st.button("Login"):
        ok, role = check_login(u, p)
        if ok: st.session_state.logged_in = True; st.session_state.username = u; st.session_state.user_role = role; st.rerun()
        else: st.error("Fail")
else:
    st.sidebar.title("🧠 HOB Brain")
    st.sidebar.info("Version 11.0 (Beta)")
    if st.sidebar.button("Logout"): st.session_state.logged_in = False; st.rerun()
    
    mp = st.sidebar.selectbox("Marketplace", ["Myntra", "Flipkart", "Ajio", "Amazon"])
    mp_cats = get_categories(mp)

    tabs = st.tabs(["🚀 Run (Smart)", "🧠 Train Brain", "🛠️ Configs"])

    # --- TAB 1: SMART RUN ---
    with tabs[0]:
        st.header(f"Generate {mp} Listings (With Learning)")
        cat = st.selectbox("Category", mp_cats)
        
        # Check if we have brain data
        if cat:
            glossary_data = get_worksheet_data(SHEET_NAME, "Visual_Glossary")
            brain_count = sum(1 for r in glossary_data if r[0] == mp and r[1] == cat)
            if brain_count > 0:
                st.success(f"🧠 Brain Active: Found {brain_count} learning examples for this category.")
            else:
                st.warning("⚠️ Brain Empty: Use 'Train Brain' tab to teach the AI.")

        infile = st.file_uploader("Upload Excel", type=["xlsx"])
        if infile and cat:
            df = pd.read_excel(infile)
            config = load_config(mp, cat)
            
            c1, c2 = st.columns(2)
            model = c1.selectbox("Model", ["GPT-4o", "Gemini 2.5 Flash"])
            mode = c2.radio("Mode", ["Test (3 rows)", "Full"], horizontal=True)
            
            # Column selector
            pot_cols = [c for c in df.columns if "url" in c.lower() or "image" in c.lower()]
            img_col = st.selectbox("Image Column", df.columns, index=df.columns.get_loc(pot_cols[0]) if pot_cols else 0)

            if st.button("Start Smart Generation"):
                df_proc = df.head(3) if "Test" in mode else df
                prog = st.progress(0); status = st.empty()
                out_rows = []
                
                # Pre-calculate Master Data Mapping
                col_master_map = {}
                if config and 'headers' in config:
                    for h in config['headers']:
                        for master_col, opts in config['master_data'].items():
                            if master_col.lower() in h.lower() or h.lower() in master_col.lower():
                                col_master_map[h] = opts
                                break

                for i, row in df_proc.iterrows():
                    # Safety Wait for Gemini
                    if "Gemini" in model: time.sleep(5)
                    
                    status.text(f"Analyzing Row {i+1}...")
                    prog.progress((i+1)/len(df_proc))
                    
                    url = str(row.get(img_col, "")).strip()
                    hints = ", ".join([f"{k}:{v}" for k,v in row.items() if k != img_col])
                    
                    # 1. CALL THE LEARNING AI
                    ai_data, err = analyze_image_with_learning(model, client, url, hints, get_seo(mp, cat), config, mp)
                    
                    new_r = row.to_dict()
                    
                    if ai_data:
                        # 2. MAP & ENFORCE
                        for h in config['headers']:
                            ai_val = None
                            if h in ai_data:
                                ai_val = ai_data[h]
                            else:
                                for k, v in ai_data.items():
                                    if k.lower() in h.lower() or h.lower() in k.lower():
                                        ai_val = v; break
                            
                            if ai_val:
                                # 3. THE POLICE CHECK
                                if h in col_master_map:
                                    clean_val = enforce_master_data(ai_val, col_master_map[h])
                                    new_r[h] = clean_val
                                else:
                                    new_r[h] = ai_val
                                    
                    out_rows.append(new_r)

                st.success("Done!")
                out_bio = BytesIO()
                pd.DataFrame(out_rows).to_excel(out_bio, index=False)
                st.download_button("Download", out_bio.getvalue(), "Smart_Result.xlsx")

    # --- TAB 2: TRAIN BRAIN ---
    with tabs[1]:
        st.header("🧠 Teach the AI")
        st.markdown("Upload examples where the AI gets confused.")
        
        t_cat = st.selectbox("Category to Teach", mp_cats, key="t_cat")
        
        if t_cat:
            conf = load_config(mp, t_cat)
            if conf and 'master_data' in conf:
                c1, c2 = st.columns(2)
                target_attr = c1.selectbox("Attribute (e.g., Pattern)", conf['master_data'].keys())
                correct_val = c2.selectbox("Correct Value", conf['master_data'][target_attr])
                
                t_url = st.text_input("Reference Image URL (Paste a link to a perfect example)")
                t_desc = st.text_input("Visual Description (e.g., 'Threads are raised above fabric surface')")
                
                if st.button("💾 Save to Brain"):
                    if t_url and t_desc:
                        if save_visual_example(mp, t_cat, target_attr, correct_val, t_url, t_desc):
                            st.success(f"Learned! Next time AI sees something like this, it will choose '{correct_val}'.")
                    else:
                        st.error("Please provide URL and Description.")
            else:
                st.error("No Master Data found for this category.")

    # --- TAB 3: CONFIGS (View Only) ---
    with tabs[2]:
        st.json(get_worksheet_data(SHEET_NAME, "Configs"))
