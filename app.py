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
from PIL import Image, ImageOps  # NEW: For Image Processing

st.set_page_config(page_title="HOB OS - Secure", layout="wide")

# ==========================================
# 1. AUTHENTICATION & DATABASE CONNECT
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_role = ""
    st.session_state.username = ""

try:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)

    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    gc = gspread.authorize(creds)
    
    SHEET_NAME = "Agency_OS_Database"
    try:
        sh = gc.open(SHEET_NAME)
        ws_configs = sh.worksheet("Configs")
        ws_seo = sh.worksheet("SEO_Data")
        try:
            ws_users = sh.worksheet("Users")
        except:
            st.error("⚠️ Database Error: 'Users' worksheet missing.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        st.stop()
    
except Exception as e:
    st.error(f"❌ Secrets Error: {str(e)}")
    st.stop()

# ==========================================
# 2. USER MANAGEMENT FUNCTIONS
# ==========================================
def check_login(username, password):
    try:
        users = ws_users.get_all_records()
        for u in users:
            if str(u['Username']).strip() == username and str(u['Password']).strip() == password:
                return True, u['Role']
        return False, None
    except: return False, None

def create_user(username, password, role):
    try:
        if ws_users.find(username): return False, "User exists"
        ws_users.append_row([username, password, role])
        return True, "Created"
    except: return False, "DB Error"

def delete_user(username):
    try:
        cell = ws_users.find(username)
        if cell: ws_users.delete_rows(cell.row); return True
        return False
    except: return False

def get_all_users(): return ws_users.get_all_records()

# ==========================================
# 3. CORE LOGIC (UPDATED V10.3)
# ==========================================
def get_categories_for_marketplace(marketplace):
    try:
        rows = ws_configs.get_all_values()
        cats = [row[1] for row in rows if len(row) > 1 and row[0] == marketplace]
        return list(set([c for c in cats if c and c != "Category"]))
    except: return []

def save_config(marketplace, category, data):
    try:
        json_str = json.dumps(data)
        rows = ws_configs.get_all_values()
        for i, row in enumerate(rows):
            if len(row) > 1 and row[0] == marketplace and row[1] == category:
                ws_configs.update_cell(i + 1, 3, json_str); return True
        ws_configs.append_row([marketplace, category, json_str]); return True
    except: return False

def load_config(marketplace, category):
    try:
        rows = ws_configs.get_all_values()
        for row in rows:
            if len(row) > 2 and row[0] == marketplace and row[1] == category:
                return json.loads(row[2])
        return None
    except: return None

def delete_config(marketplace, category):
    try:
        rows = ws_configs.get_all_values()
        for i, row in enumerate(rows):
            if len(row) > 1 and row[0] == marketplace and row[1] == category:
                ws_configs.delete_rows(i + 1); return True
        return False
    except: return False

def save_seo(marketplace, category, keywords_list):
    try:
        kw_string = ", ".join([str(k).strip() for k in keywords_list if str(k).strip()])
        rows = ws_seo.get_all_values()
        for i, row in enumerate(rows):
            if len(row) > 1 and row[0] == marketplace and row[1] == category:
                ws_seo.update_cell(i + 1, 3, kw_string); return True
        ws_seo.append_row([marketplace, category, kw_string]); return True
    except: return False

def get_seo(marketplace, category):
    try:
        rows = ws_seo.get_all_values()
        for row in rows:
            if len(row) > 2 and row[0] == marketplace and row[1] == category:
                return row[2]
        return ""
    except: return ""

def parse_master_data(file):
    df = pd.read_excel(file)
    valid_options = {}
    for col in df.columns:
        options = df[col].dropna().astype(str).unique().tolist()
        if len(options) > 0: valid_options[col] = options
    return valid_options

def encode_image_from_url(url):
    try:
        if pd.isna(url) or str(url).strip() == "": return None, "Empty URL"
        if "dropbox.com" in url:
            url = url.replace("?dl=0", "").replace("&dl=0", "") + ("&dl=1" if "?" in url else "?dl=1")
        response = requests.get(url, timeout=10)
        return (base64.b64encode(response.content).decode('utf-8'), None) if response.status_code == 200 else (None, "Download Error")
    except Exception as e: return None, str(e)

# --- NEW: IMAGE PROCESSING UTILITY ---
def process_image_to_square(image_file, size=1000):
    try:
        img = Image.open(image_file)
        if img.mode in ('RGBA', 'LA'):
            background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
            background.paste(img, img.split()[-1])
            img = background
        img = img.convert("RGB")
        
        # Calculate aspect ratio preserving resize
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        
        # Create white canvas
        new_img = Image.new("RGB", (size, size), (255, 255, 255))
        
        # Paste centered
        left = (size - img.width) // 2
        top = (size - img.height) // 2
        new_img.paste(img, (left, top))
        
        return new_img
    except Exception as e:
        return None

# --- UPDATED: AMAZON AWARE AI ---
def analyze_image_configured(client, image_url, user_hints, keywords, config, marketplace):
    base64_image, error = encode_image_from_url(image_url)
    if error: return None, error

    relevant_options = {}
    ai_target_headers = []
    
    for col, settings in config['column_mapping'].items():
        if settings['source'] == 'AI':
            ai_target_headers.append(col)
            for master_col, opts in config['master_data'].items():
                if master_col.lower() in col.lower() or col.lower() in master_col.lower():
                    relevant_options[col] = opts
                    break

    seo_instruction = ""
    if keywords: seo_instruction = f"MANDATORY SEO KEYWORDS: {keywords}"

    # --- MARKETPLACE SPECIFIC LOGIC ---
    mp_instruction = ""
    if marketplace.lower() == "amazon":
        mp_instruction = """
        AMAZON SPECIFIC RULES:
        1. BULLET POINTS: If output keys contain 'Bullet' or 'Feature', write 5 distinct selling points (Material, Fit, Usage, Care, Design). Start each with a capitalized phrase (e.g., 'PREMIUM COTTON:').
        2. SEARCH TERMS: If keys contain 'Search Terms', provide 250 bytes of unique keywords, no commas.
        3. TITLE: Format: [Brand] + [Department] + [Material] + [Style] + [Color] (Max 200 chars).
        """
    elif marketplace.lower() == "myntra":
        mp_instruction = "MYNTRA RULES: Focus on fashion attributes (Neck, Sleeve, Pattern). Title should be short and catchy."
    
    prompt = f"""
    You are a Cataloging Expert for {marketplace}.
    CATEGORY: {config['category_name']}
    CONTEXT: {user_hints}
    TASK: Fill these attributes: {ai_target_headers}
    {seo_instruction}
    
    {mp_instruction}
    
    STRICT DROPDOWN VALUES (Must Match Exactly):
    {json.dumps(relevant_options, indent=2)}
    
    RETURN RAW JSON.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a JSON-only assistant."},
                {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
            ],
            response_format={"type": "json_object"},
            max_tokens=2000
        )
        return json.loads(response.choices[0].message.content), None
    except Exception as e:
        return None, str(e)


# ==========================================
# 4. MAIN APP LOGIC
# ==========================================

if not st.session_state.logged_in:
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.title("🔒 HOB OS Login")
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                is_valid, role = check_login(user, pwd)
                if is_valid:
                    st.session_state.logged_in = True; st.session_state.username = user; st.session_state.user_role = role
                    st.rerun()
                else: st.error("Invalid Credentials")
else:
    st.sidebar.title("🌍 HOB OS")
    st.sidebar.caption(f"User: {st.session_state.username} ({st.session_state.user_role})")
    if st.sidebar.button("Log Out"): st.session_state.logged_in = False; st.rerun()
    st.sidebar.divider()

    selected_mp = st.sidebar.selectbox("Marketplace", ["Myntra", "Flipkart", "Ajio", "Amazon", "Nykaa"])
    if st.sidebar.button("🔄 Refresh Data"): st.rerun()

    mp_cats = get_categories_for_marketplace(selected_mp)
    st.sidebar.write(f"{len(mp_cats)} Categories Found")
    
    # --- TAB DEFINITIONS ---
    # Standard: Setup, SEO, Run, Tools
    # Admin: + Configs, Admin
    base_tabs = ["🛠️ Setup", "📈 SEO", "🚀 Run", "🖼️ Tools"]
    if st.session_state.user_role.lower() == "admin":
        base_tabs += ["🗑️ Configs", "👥 Admin"]
        
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
                        for col, rule in loaded['column_mapping'].items():
                            src_map = {"AI": "AI Generation", "INPUT": "Input Excel", "FIXED": "Fixed Value", "BLANK": "Leave Blank"}
                            default_mapping.append({"Column Name": col, "Source": src_map.get(rule['source'], "Leave Blank"), "Fixed Value (If Fixed)": rule['value']})
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
                    default_mapping.append({"Column Name": h, "Source": src, "Fixed Value (If Fixed)": ""})

            edited_df = st.data_editor(pd.DataFrame(default_mapping), column_config={"Source": st.column_config.SelectboxColumn("Source", options=["Input Excel", "AI Generation", "Fixed Value", "Leave Blank"])}, hide_index=True, use_container_width=True, height=400)
            
            if st.button("Save Config"):
                final_map = {}
                for i, row in edited_df.iterrows():
                    src_code = "AI" if row['Source'] == "AI Generation" else "INPUT" if row['Source'] == "Input Excel" else "FIXED" if row['Source'] == "Fixed Value" else "BLANK"
                    final_map[row['Column Name']] = {"source": src_code, "value": row['Fixed Value (If Fixed)']}
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

    # --- TAB 3: RUN ---
    with tabs[2]:
        st.header(f"3. Run {selected_mp} Generator")
        if not mp_cats: st.warning("No categories configured yet."); st.stop()
        
        run_cat = st.selectbox("Select Category", mp_cats, key="run")
        if run_cat:
            config = load_config(selected_mp, run_cat)
            if config:
                required_cols = ["Image URL"] + [col for col, rule in config.get('column_mapping', {}).items() if rule.get('source') == 'INPUT']
                st.info(f"ℹ️ Required Input Columns: {', '.join(required_cols)}")
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer: pd.DataFrame(columns=required_cols).to_excel(writer, index=False)
                st.download_button("📥 Download Input Template", output.getvalue(), file_name=f"{selected_mp}_{run_cat}_Template.xlsx")

        active_kws = get_seo(selected_mp, run_cat)
        input_file = st.file_uploader("Upload Input Data (filled template)", type=["xlsx"], key="run_in")
        
        if input_file:
            df_input = pd.read_excel(input_file)
            row_count = len(df_input)
            st.metric("📦 SKUs to Process", row_count)
            
            if st.button("▶️ Start Generation"):
                img_col = next((c for c in df_input.columns if "front" in c.lower() or "image" in c.lower() or "url" in c.lower()), None)
                if not img_col: st.error("❌ No 'Image URL' column found."); st.stop()
                
                progress = st.progress(0); status = st.empty()
                final_rows = []; cache = {}; mapping = config['column_mapping']
                
                for idx, row in df_input.iterrows():
                    status.text(f"Processing {idx+1}/{row_count}")
                    progress.progress((idx+1)/row_count)
                    
                    img_url = str(row.get(img_col, "")).strip()
                    ai_data = None
                    needs_ai = any(m['source']=='AI' for m in mapping.values())
                    
                    if needs_ai:
                        if img_url in cache: ai_data = cache[img_url]
                        else:
                            hints = ", ".join([f"{k}: {v}" for k,v in row.items() if str(v) != "nan" and k != img_col])
                            ai_data, _ = analyze_image_configured(client, img_url, hints, active_kws, config, selected_mp)
                            if ai_data: cache[img_url] = ai_data
                    
                    new_row = {}
                    for col in config['headers']:
                        rule = mapping.get(col, {'source': 'BLANK'})
                        if rule['source'] == 'INPUT':
                            val = ""
                            if col in df_input.columns: val = row[col]
                            else:
                                for ic in df_input.columns:
                                    if ic.lower() in col.lower(): val = row[ic]; break
                            new_row[col] = val
                        elif rule['source'] == 'FIXED': new_row[col] = rule['value']
                        elif rule['source'] == 'AI' and ai_data:
                            found = False
                            if col in ai_data: new_row[col] = ai_data[col]; found = True
                            else:
                                for k,v in ai_data.items():
                                    if k.lower().replace(" ","") in col.lower().replace(" ",""): new_row[col] = v; found = True; break
                            if not found: new_row[col] = ""
                        else: new_row[col] = ""
                    final_rows.append(new_row)
                
                output_gen = BytesIO()
                with pd.ExcelWriter(output_gen, engine='xlsxwriter') as writer: pd.DataFrame(final_rows).to_excel(writer, index=False)
                st.success("✅ Done!")
                st.download_button("⬇️ Download Catalog", output_gen.getvalue(), file_name=f"{selected_mp}_{run_cat}_Generated.xlsx")

    # --- TAB 4: TOOLS (NEW V10.3) ---
    with tabs[3]:
        st.header("🖼️ Bulk Image Resizer")
        st.caption("Convert images to Marketplace Standards (Square, White Background, 1000px).")
        
        tool_files = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg", "webp"], accept_multiple_files=True)
        target_size = st.number_input("Target Size (px)", value=1000, min_value=500, max_value=3000)
        
        if tool_files and st.button("Process Images"):
            zip_buffer = BytesIO()
            prog_bar = st.progress(0)
            
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for i, f in enumerate(tool_files):
                    processed = process_image_to_square(f, target_size)
                    if processed:
                        # Save to memory buffer
                        img_byte_arr = BytesIO()
                        processed.save(img_byte_arr, format='JPEG', quality=90)
                        # Add to zip
                        fname = f.name.rsplit('.', 1)[0] + "_processed.jpg"
                        zf.writestr(fname, img_byte_arr.getvalue())
                    prog_bar.progress((i+1)/len(tool_files))
            
            st.success(f"Processed {len(tool_files)} images!")
            st.download_button("⬇️ Download ZIP", zip_buffer.getvalue(), file_name="Marketplace_Images.zip", mime="application/zip")

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
