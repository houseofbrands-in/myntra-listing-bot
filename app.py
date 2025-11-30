import streamlit as st
import pandas as pd
import json
import base64
import requests
from openai import OpenAI
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Agency OS (Multi-Marketplace)", layout="wide")

# --- AUTHENTICATION ---
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
    except Exception as e:
        st.error(f"❌ Database Error: {str(e)}")
        st.stop()
    
except Exception as e:
    st.error(f"❌ Connection Error: {str(e)}")
    st.stop()

# ================= DATABASE FUNCTIONS (UPDATED FOR MARKETPLACE) =================

def get_categories_for_marketplace(marketplace):
    try:
        # Get all rows
        rows = ws_configs.get_all_values()
        # Skip header if exists, but we'll just filter
        # Col A = Marketplace, Col B = Category
        cats = [row[1] for row in rows if len(row) > 1 and row[0] == marketplace]
        # Remove duplicates and empty
        return list(set([c for c in cats if c and c != "Category"]))
    except:
        return []

def save_config(marketplace, category, data):
    try:
        json_str = json.dumps(data)
        rows = ws_configs.get_all_values()
        
        # Look for existing row to update
        cell_row = None
        for i, row in enumerate(rows):
            if len(row) > 1 and row[0] == marketplace and row[1] == category:
                cell_row = i + 1 # 1-based index
                break
        
        if cell_row:
            # Update Column C (Index 3)
            ws_configs.update_cell(cell_row, 3, json_str)
        else:
            # Append new row: [Marketplace, Category, JSON]
            ws_configs.append_row([marketplace, category, json_str])
        return True
    except Exception as e:
        st.error(f"Save Failed: {e}")
        return False

def load_config(marketplace, category):
    try:
        rows = ws_configs.get_all_values()
        for row in rows:
            if len(row) > 2 and row[0] == marketplace and row[1] == category:
                return json.loads(row[2])
        return None
    except:
        return None

def delete_config(marketplace, category):
    try:
        rows = ws_configs.get_all_values()
        for i, row in enumerate(rows):
            if len(row) > 1 and row[0] == marketplace and row[1] == category:
                ws_configs.delete_rows(i + 1)
                return True
        return False
    except:
        return False

# --- SEO Functions (Updated) ---
def save_seo(marketplace, category, keywords_list):
    try:
        kw_string = ", ".join([str(k).strip() for k in keywords_list if str(k).strip()])
        rows = ws_seo.get_all_values()
        
        cell_row = None
        for i, row in enumerate(rows):
            if len(row) > 1 and row[0] == marketplace and row[1] == category:
                cell_row = i + 1
                break
        
        if cell_row:
            ws_seo.update_cell(cell_row, 3, kw_string)
        else:
            ws_seo.append_row([marketplace, category, kw_string])
        return True
    except Exception as e: return False

def get_seo(marketplace, category):
    try:
        rows = ws_seo.get_all_values()
        for row in rows:
            if len(row) > 2 and row[0] == marketplace and row[1] == category:
                return row[2]
        return ""
    except: return ""

# ================= HELPER FUNCTIONS =================
def parse_master_data(file):
    df = pd.read_excel(file)
    valid_options = {}
    for col in df.columns:
        options = df[col].dropna().astype(str).unique().tolist()
        if len(options) > 0:
            valid_options[col] = options
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

# ================= CORE AI LOGIC =================
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
    if keywords:
        seo_instruction = f"MANDATORY KEYWORDS TO USE: {keywords}"

    prompt = f"""
    You are a Cataloging Expert for {marketplace}.
    CATEGORY: {config['category_name']}
    USER CONTEXT: {user_hints}
    
    TASK: Fill these attributes: {ai_target_headers}
    
    {seo_instruction}

    ---------------------------------------------------------
    STRICT DROPDOWN VALUES (Must Match Exactly):
    {json.dumps(relevant_options, indent=2)}

    ---------------------------------------------------------
    CREATIVE RULES FOR {marketplace.upper()}:
    1. Title: Standard {marketplace} formula (Brand + Attributes).
    2. Description: engaging, bullet points if possible.
    3. Keywords: High traffic search terms.

    RETURN ONLY RAW JSON.
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

# ================= APP UI =================

# --- SIDEBAR: GLOBAL MARKETPLACE SELECTOR ---
st.sidebar.title("🌍 Agency OS")
st.sidebar.caption("Project M - Version 9.0")
selected_mp = st.sidebar.selectbox("Select Marketplace", ["Myntra", "Flipkart", "Ajio", "Amazon", "Nykaa"])
st.sidebar.divider()

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# Get categories for THIS marketplace only
mp_cats = get_categories_for_marketplace(selected_mp)
st.sidebar.write(f"Found {len(mp_cats)} categories for {selected_mp}")


tab1, tab2, tab3, tab4 = st.tabs(["🛠️ Rules & Setup", "📈 SEO Keywords", "🚀 Run Generator", "🗑️ Manage"])

# ---------------- TAB 1: SETUP (Updated with Logic Validator) ----------------
with tab1:
    st.header(f"Step 1: Setup {selected_mp} Rules")
    
    mode = st.radio("Mode", ["Create New", "Edit Existing"], horizontal=True)
    cat_name = ""
    headers = []
    master_options = {}
    default_mapping = []

    if mode == "Edit Existing":
        if mp_cats:
            edit_cat = st.selectbox(f"Select {selected_mp} Category", mp_cats)
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
            st.warning(f"No {selected_mp} categories yet.")
    else:
        cat_name = st.text_input(f"New {selected_mp} Category Name")

    c1, c2 = st.columns(2)
    template_file = c1.file_uploader("1. Upload Template (.xlsx)", type=["xlsx"])
    master_file = c2.file_uploader("2. Upload Master Data (.xlsx)", type=["xlsx"])

    if template_file: headers = pd.read_excel(template_file).columns.tolist()
    if master_file: master_options = parse_master_data(master_file)

    if headers:
        st.divider()
        st.subheader("Map Your Columns")
        if not default_mapping:
            for h in headers: 
                # Smart Auto-Mapping
                src = "Leave Blank"
                if "image" in h.lower() or "sku" in h.lower(): src = "Input Excel"
                elif h in master_options: src = "AI Generation" # Auto-detect strict
                elif "name" in h.lower() or "description" in h.lower(): src = "AI Generation" # Auto-detect creative
                
                default_mapping.append({"Column Name": h, "Source": src, "Fixed Value (If Fixed)": ""})

        edited_df = st.data_editor(
            pd.DataFrame(default_mapping), 
            column_config={
                "Source": st.column_config.SelectboxColumn("Source", options=["Input Excel", "AI Generation", "Fixed Value", "Leave Blank"], required=True)
            }, 
            hide_index=True, 
            use_container_width=True,
            height=400
        )
        
        # --- NEW: LOGIC VALIDATOR ---
        st.divider()
        st.subheader("🕵️ AI Logic Preview")
        
        strict_cols = []
        creative_cols = []
        
        for index, row in edited_df.iterrows():
            if row['Source'] == "AI Generation":
                col_name = row['Column Name']
                # Check if this column exists in the uploaded Master Data (Fuzzy match)
                found_in_master = False
                for m_col in master_options:
                    if m_col.lower() in col_name.lower() or col_name.lower() in m_col.lower():
                        found_in_master = True
                        break
                
                if found_in_master:
                    strict_cols.append(col_name)
                else:
                    creative_cols.append(col_name)
        
        c_val1, c_val2 = st.columns(2)
        with c_val1:
            if strict_cols:
                st.success(f"🔒 **Strict Mode (Dropdowns):**\n\nThe AI will choose strictly from your Master Data for:\n\n" + ", ".join([f"`{c}`" for c in strict_cols]))
            else:
                st.info("No Strict Columns detected.")
                
        with c_val2:
            if creative_cols:
                st.warning(f"✨ **Creative Mode (Writing):**\n\nThe AI will generate creative text/SEO for:\n\n" + ", ".join([f"`{c}`" for c in creative_cols]))
            else:
                st.info("No Creative Columns detected.")
        # -----------------------------

        st.divider()
        if st.button("Save Configuration"):
            if not cat_name:
                st.error("Please enter a Category Name.")
            else:
                final_map = {}
                for i, row in edited_df.iterrows():
                    src_code = "AI" if row['Source'] == "AI Generation" else "INPUT" if row['Source'] == "Input Excel" else "FIXED" if row['Source'] == "Fixed Value" else "BLANK"
                    final_map[row['Column Name']] = {"source": src_code, "value": row['Fixed Value (If Fixed)']}
                
                payload = {"category_name": cat_name, "headers": headers, "master_data": master_options, "column_mapping": final_map}
                
                if save_config(selected_mp, cat_name, payload):
                    st.success(f"✅ Saved {cat_name} for {selected_mp}!"); time.sleep(1); st.rerun()
# ---------------- TAB 2: SEO ----------------
with tab2:
    st.header(f"Step 2: {selected_mp} Keywords")
    
    if not mp_cats:
        st.warning("Create a category first.")
    else:
        seo_cat = st.selectbox("Select Category", mp_cats, key="seo_cat")
        curr_kw = get_seo(selected_mp, seo_cat)
        
        if curr_kw:
            st.info(f"Active Keywords: {curr_kw[:100]}...")
        else:
            st.warning("No keywords set.")

        kw_file = st.file_uploader("Upload Keywords File (Column A)", type=["xlsx", "csv"])
        if kw_file and st.button("Update Keywords"):
            try:
                df_kw = pd.read_csv(kw_file) if kw_file.name.endswith('.csv') else pd.read_excel(kw_file)
                kw_list = df_kw.iloc[:, 0].dropna().astype(str).tolist()
                if save_seo(selected_mp, seo_cat, kw_list):
                    st.success("Updated!"); time.sleep(1); st.rerun()
            except Exception as e: st.error(str(e))

# ---------------- TAB 3: RUNNER ----------------
with tab3:
    st.header(f"Step 3: Generate {selected_mp} Listings")
    
    if not mp_cats: st.warning("No categories found."); st.stop()
    
    run_cat = st.selectbox("Select Category", mp_cats, key="run_cat")
    active_kws = get_seo(selected_mp, run_cat)
    
    if active_kws: st.caption("✨ Optimizing with custom keywords")
    else: st.caption("⚠️ Using generic AI SEO")
    
    input_file = st.file_uploader("Upload Input Data (.xlsx)", type=["xlsx"], key="run_in")
    
    if input_file and st.button("▶️ Generate Catalog"):
        config = load_config(selected_mp, run_cat)
        df_input = pd.read_excel(input_file)
        
        # Flexible image column search
        img_col = next((c for c in df_input.columns if "front" in c.lower() or "image" in c.lower() or "url" in c.lower()), None)
        if not img_col: st.error("No 'Image' column found."); st.stop()
        
        progress = st.progress(0); status = st.empty()
        final_rows = []
        cache = {}
        mapping = config['column_mapping']
        
        for idx, row in df_input.iterrows():
            status.text(f"Processing Row {idx+1}")
            progress.progress((idx+1)/len(df_input))
            
            img_url = str(row.get(img_col, "")).strip()
            ai_data = None
            needs_ai = any(m['source']=='AI' for m in mapping.values())
            
            if needs_ai:
                if img_url in cache: ai_data = cache[img_url]
                else:
                    # Generic hints extraction
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
                        # Fuzzy match input columns
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
        with open(fn, "rb") as f: st.download_button("Download Excel", f, file_name=fn)

# ---------------- TAB 4: MANAGE ----------------
with tab4:
    st.header(f"Manage {selected_mp} Configs")
    to_del = st.selectbox("Delete Category", [""]+mp_cats)
    if to_del and st.button("Delete"):
        delete_config(selected_mp, to_del)
        st.success("Deleted"); time.sleep(1); st.rerun()

