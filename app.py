import streamlit as st
import pandas as pd
import json
import base64
import requests
from openai import OpenAI
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Agency OS - Secure", layout="wide")

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
        # NEW: Users Sheet
        try:
            ws_users = sh.worksheet("Users")
        except:
            st.error("⚠️ Database Error: Please create a worksheet named 'Users' with cols: Username, Password, Role")
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
            # Convert everything to string to be safe
            if str(u['Username']).strip() == username and str(u['Password']).strip() == password:
                return True, u['Role']
        return False, None
    except Exception as e:
        st.error(f"Login Logic Error: {e}")
        return False, None

def create_user(username, password, role):
    try:
        # Check if exists
        cell = ws_users.find(username)
        if cell: return False, "User already exists"
        ws_users.append_row([username, password, role])
        return True, "User created"
    except: return False, "Database error"

def delete_user(username):
    try:
        cell = ws_users.find(username)
        if cell:
            ws_users.delete_rows(cell.row)
            return True
        return False
    except: return False

def get_all_users():
    return ws_users.get_all_records()

# ==========================================
# 3. CORE LOGIC (V9.2 Functions)
# ==========================================
# ... (Keeping previous logic functions exactly same) ...

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
        cell_row = None
        for i, row in enumerate(rows):
            if len(row) > 1 and row[0] == marketplace and row[1] == category:
                cell_row = i + 1; break
        if cell_row: ws_configs.update_cell(cell_row, 3, json_str)
        else: ws_configs.append_row([marketplace, category, json_str])
        return True
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
        cell_row = None
        for i, row in enumerate(rows):
            if len(row) > 1 and row[0] == marketplace and row[1] == category:
                cell_row = i + 1; break
        if cell_row: ws_seo.update_cell(cell_row, 3, kw_string)
        else: ws_seo.append_row([marketplace, category, kw_string])
        return True
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
    except Exception as e:
        return None, str(e)

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

    prompt = f"""
    You are a Cataloging Expert for {marketplace}.
    CATEGORY: {config['category_name']}
    CONTEXT: {user_hints}
    TASK: Fill these attributes: {ai_target_headers}
    {seo_instruction}
    
    STRICT DROPDOWN VALUES (Must Match Exactly):
    {json.dumps(relevant_options, indent=2)}
    
    CREATIVE RULES:
    1. Title: Standard {marketplace} formula.
    2. Description: Engaging, include keywords.
    3. Keywords: High traffic search terms.
    
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
# 4. MAIN APP LOGIC (LOGIN WRAPPER)
# ==========================================

if not st.session_state.logged_in:
    # --- LOGIN SCREEN ---
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.title("🔒 Agency OS Login")
        st.write("Please sign in to access the tool.")
        
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                is_valid, role = check_login(user, pwd)
                if is_valid:
                    st.session_state.logged_in = True
                    st.session_state.username = user
                    st.session_state.user_role = role
                    st.success("Login Successful!")
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")

else:
    # --- LOGGED IN DASHBOARD ---
    
    # Sidebar Profile
    st.sidebar.title("🌍 Agency OS")
    st.sidebar.caption(f"User: {st.session_state.username} ({st.session_state.user_role})")
    
    if st.sidebar.button("Log Out"):
        st.session_state.logged_in = False
        st.rerun()
        
    st.sidebar.divider()

    # Marketplace Selector
    selected_mp = st.sidebar.selectbox("Marketplace", ["Myntra", "Flipkart", "Ajio", "Amazon", "Nykaa"])
    st.sidebar.divider()
    
    # Refresh Button
    if st.sidebar.button("🔄 Refresh Data"): st.rerun()

    # Get Categories
    mp_cats = get_categories_for_marketplace(selected_mp)
    st.sidebar.write(f"{len(mp_cats)} Categories Found")
    
    # --- ROLE BASED TABS ---
    
    # Define available tabs based on role
    if st.session_state.user_role.lower() == "admin":
        tab_names = ["🛠️ Setup", "📈 SEO", "🚀 Run", "🗑️ Configs", "👥 Admin Console"]
    else:
        # Standard User
        tab_names = ["🛠️ Setup", "📈 SEO", "🚀 Run"]
        
    tabs = st.tabs(tab_names)

    # --- TAB 1: SETUP ---
    with tabs[0]:
        st.header(f"1. Setup {selected_mp} Rules")
        mode = st.radio("Mode", ["Create New", "Edit Existing"], horizontal=True)
        cat_name = ""
        headers = []
        master_options = {}
        default_mapping = []

        if mode == "Edit Existing":
            if mp_cats:
                edit_cat = st.selectbox(f"Select Category", mp_cats)
                if edit_cat:
                    loaded = load_config(selected_mp, edit_cat)
                    if loaded:
                        cat_name = loaded['category_name']
                        headers = loaded['headers']
                        master_options = loaded['master_data']
                        for col, rule in loaded['column_mapping'].items():
                            src_map = {"AI": "AI Generation", "INPUT": "Input Excel", "FIXED": "Fixed Value", "BLANK": "Leave Blank"}
                            default_mapping.append({"Column Name": col, "Source": src_map.get(rule['source'], "Leave Blank"), "Fixed Value (If Fixed)": rule['value']})
        else:
            cat_name = st.text_input(f"New {selected_mp} Category Name")

        c1, c2 = st.columns(2)
        template_file = c1.file_uploader("Template (.xlsx)", type=["xlsx"])
        master_file = c2.file_uploader("Master Data (.xlsx)", type=["xlsx"])

        if template_file: headers = pd.read_excel(template_file).columns.tolist()
        if master_file: master_options = parse_master_data(master_file)

        if headers:
            st.divider()
            if not default_mapping:
                for h in headers:
                    src = "Leave Blank"
                    h_low = h.lower()
                    if "image" in h_low or "sku" in h_low: src = "Input Excel"
                    elif h in master_options or "name" in h_low or "desc" in h_low: src = "AI Generation"
                    default_mapping.append({"Column Name": h, "Source": src, "Fixed Value (If Fixed)": ""})

            edited_df = st.data_editor(
                pd.DataFrame(default_mapping), 
                column_config={"Source": st.column_config.SelectboxColumn("Source", options=["Input Excel", "AI Generation", "Fixed Value", "Leave Blank"])}, 
                hide_index=True, 
                use_container_width=True, 
                height=400
            )
            
            # --- LOGIC VALIDATOR (FIXED) ---
            st.divider()
            st.subheader("🕵️ AI Logic Validation")
            
            strict, creative = [], []
            for i, row in edited_df.iterrows():
                if row['Source'] == "AI Generation":
                    # Check if column exists in master options (Fuzzy Match)
                    # FIX: Ensure we don't match generic terms like "Name" to "Product Name" incorrectly if "Name" isn't in master
                    found = False
                    col_name = row['Column Name']
                    for m_col in master_options:
                        if m_col.lower() == col_name.lower(): # Exact match preferred
                            found = True; break
                        elif m_col.lower() in col_name.lower() and len(m_col) > 3: # Substring match only if master col is specific length
                            found = True; break
                    
                    if found: strict.append(col_name)
                    else: creative.append(col_name)
            
            c_v1, c_v2 = st.columns(2)
            with c_v1:
                st.success(f"🔒 **Strict Mode** ({len(strict)})")
                with st.expander("View Dropdown Columns"):
                    st.write(", ".join([f"`{c}`" for c in strict]))
                    
            with c_v2:
                st.warning(f"✨ **Creative Mode** ({len(creative)})")
                with st.expander("View Content Columns"):
                    st.write(", ".join([f"`{c}`" for c in creative]))
            
            st.caption("ℹ️ *Strict columns force the AI to pick from your Master Data. Creative columns allow the AI to write freely.*")
            # -------------------------------

            if st.button("Save Config"):
                final_map = {}
                for i, row in edited_df.iterrows():
                    src_code = "AI" if row['Source'] == "AI Generation" else "INPUT" if row['Source'] == "Input Excel" else "FIXED" if row['Source'] == "Fixed Value" else "BLANK"
                    final_map[row['Column Name']] = {"source": src_code, "value": row['Fixed Value (If Fixed)']}
                
                payload = {"category_name": cat_name, "headers": headers, "master_data": master_options, "column_mapping": final_map}
                if save_config(selected_mp, cat_name, payload):
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
        if not mp_cats: st.warning("No categories."); st.stop()
        
        run_cat = st.selectbox("Select Category", mp_cats, key="run")
        active_kws = get_seo(selected_mp, run_cat)
        input_file = st.file_uploader("Input Data (.xlsx)", type=["xlsx"], key="run_in")
        
        if input_file:
            df_input = pd.read_excel(input_file)
            row_count = len(df_input)
            
            est_cost_usd = row_count * 0.02
            
            st.divider()
            c_cost1, c_cost2 = st.columns(2)
            c_cost1.metric("📦 Items", f"{row_count} SKUs")
            c_cost2.metric("💲 Est. Cost", f"${est_cost_usd:.2f}")
            
            if st.button("▶️ Start Generation"):
                config = load_config(selected_mp, run_cat)
                img_col = next((c for c in df_input.columns if "front" in c.lower() or "image" in c.lower() or "url" in c.lower()), None)
                if not img_col: st.error("No Image column."); st.stop()
                
                progress = st.progress(0); status = st.empty()
                final_rows = []
                cache = {}
                mapping = config['column_mapping']
                
                for idx, row in df_input.iterrows():
                    status.text(f"Processing {idx+1}/{row_count}")
                    progress.progress((idx+1)/row_count)
                    
                    img_url = str(row.get(img_col, "")).strip()
                    ai_data = None
                    needs_ai = any(m['source']=='AI' for m in mapping.values())
                    
                    if needs_ai:
                        if img_url in cache: ai_data = cache[img_url]
                        else:
                            hints = ", ".join([f"{k}: {v}" for k,v in row.items() if "color" in k.lower() or "fabric" in k.lower()])
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
                    
                out_df = pd.DataFrame(final_rows)
                fn = f"{selected_mp}_{run_cat}_Generated.xlsx"
                out_df.to_excel(fn, index=False)
                st.success("Done!")
                with open(fn, "rb") as f: st.download_button("Download", f, file_name=fn)

    # --- ADMIN ONLY TABS ---
    if st.session_state.user_role.lower() == "admin":
        
        # Tab 4: Manage Configs
        with tabs[3]:
            st.header("Manage Configs")
            to_del = st.selectbox("Delete", [""]+mp_cats)
            if to_del and st.button("Delete Config"):
                delete_config(selected_mp, to_del)
                st.success("Deleted"); time.sleep(1); st.rerun()

        # Tab 5: Admin Console (User Management)
        with tabs[4]:
            st.header("👥 Admin Console: User Management")
            
            # View Users
            all_users = get_all_users()
            if all_users:
                st.dataframe(pd.DataFrame(all_users))
            
            st.divider()
            
            # Create User
            c_add1, c_add2 = st.columns(2)
            with c_add1:
                st.subheader("Add New User")
                with st.form("add_user"):
                    new_u = st.text_input("Username")
                    new_p = st.text_input("Password")
                    new_r = st.selectbox("Role", ["user", "admin"])
                    if st.form_submit_button("Create User"):
                        ok, msg = create_user(new_u, new_p, new_r)
                        if ok: st.success(msg); time.sleep(1); st.rerun()
                        else: st.error(msg)
            
            # Delete User
            with c_add2:
                st.subheader("Remove User")
                u_to_del = st.selectbox("Select User to Remove", [u['Username'] for u in all_users if str(u['Username']) != "admin"])
                if st.button("Delete User"):
                    if delete_user(u_to_del):
                        st.success(f"Removed {u_to_del}"); time.sleep(1); st.rerun()
                    else: st.error("Failed")

