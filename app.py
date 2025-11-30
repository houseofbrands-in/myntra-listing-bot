import streamlit as st
import pandas as pd
import json
import base64
import requests
from openai import OpenAI
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Myntra Agency OS (Cloud)", layout="wide")

# --- AUTHENTICATION ---
try:
    # 1. OpenAI Key
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)

    # 2. Google Sheets Connection
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    gc = gspread.authorize(creds)
    
    # CONNECT TO SHEETS
    SHEET_NAME = "Agency_OS_Database"
    try:
        sh = gc.open(SHEET_NAME)
        ws_configs = sh.worksheet("Configs")
        # Try to open SEO sheet, if not exist, warn user
        try:
            ws_seo = sh.worksheet("SEO_Data")
        except:
            st.error("⚠️ Missing Worksheet: Please create a new tab named 'SEO_Data' in your Google Sheet.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Spreadsheet Error: {str(e)}")
        st.stop()
    
except Exception as e:
    st.error(f"❌ Connection Error: {str(e)}")
    st.stop()

# ================= DATABASE FUNCTIONS =================

def get_all_categories():
    try:
        cats = ws_configs.col_values(1)[1:] 
        return [c for c in cats if c] 
    except:
        return []

# --- Config Handling ---
def save_config_to_sheet(name, data):
    try:
        json_str = json.dumps(data)
        cell = None
        try: cell = ws_configs.find(name)
        except: pass
        if cell: ws_configs.update_cell(cell.row, 2, json_str)
        else: ws_configs.append_row([name, json_str])
        return True
    except Exception as e: return False

def load_config_from_sheet(name):
    try:
        cell = ws_configs.find(name)
        json_str = ws_configs.cell(cell.row, 2).value
        return json.loads(json_str)
    except: return None

def delete_config_from_sheet(name):
    try:
        cell = ws_configs.find(name)
        ws_configs.delete_rows(cell.row)
        return True
    except: return False

# --- SEO/Keyword Handling (New) ---
def save_keywords_to_sheet(category, keywords_list):
    try:
        # Join list into a single string for storage
        kw_string = ", ".join([str(k).strip() for k in keywords_list if str(k).strip()])
        
        cell = None
        try: cell = ws_seo.find(category)
        except: pass
        
        if cell:
            # Update existing
            ws_seo.update_cell(cell.row, 2, kw_string)
        else:
            # Create new
            ws_seo.append_row([category, kw_string])
        return True
    except Exception as e:
        st.error(f"SEO Save Error: {e}")
        return False

def get_keywords_for_category(category):
    try:
        cell = ws_seo.find(category)
        # Return string
        return ws_seo.cell(cell.row, 2).value
    except:
        return "" # No keywords found

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

# ================= CORE AI LOGIC (UPDATED WITH KEYWORDS) =================
def analyze_image_configured(client, image_url, user_hints, keywords, config):
    base64_image, error = encode_image_from_url(image_url)
    if error: return None, error

    # 1. Prepare Valid Options (Strict Mode)
    relevant_options = {}
    ai_target_headers = []
    for col, settings in config['column_mapping'].items():
        if settings['source'] == 'AI':
            ai_target_headers.append(col)
            for master_col, opts in config['master_data'].items():
                if master_col.lower() in col.lower() or col.lower() in master_col.lower():
                    relevant_options[col] = opts
                    break

    # 2. Inject Keywords into Prompt
    seo_instruction = ""
    if keywords:
        seo_instruction = f"""
        CRITICAL SEO INSTRUCTIONS:
        The user has provided specific high-performing keywords. 
        You MUST integrate the following terms naturally into the 'Product Name', 'Description', and 'Tags':
        KEYWORDS TO USE: {keywords}
        """

    prompt = f"""
    You are a Myntra/Flipkart Cataloging Expert.
    CATEGORY: {config['category_name']}
    USER HINTS: {user_hints}
    
    TASK: Analyze the image and fill these attributes: {ai_target_headers}
    
    {seo_instruction}

    ---------------------------------------------------------
    SECTION 1: TECHNICAL ATTRIBUTES (Strict Dropdown Choice)
    ---------------------------------------------------------
    For technical fields (Fabric, Color, Neck, etc.), choose ONLY from here:
    {json.dumps(relevant_options, indent=2)}

    ---------------------------------------------------------
    SECTION 2: CREATIVE CONTENT (High Discovery)
    ---------------------------------------------------------
    1. Product Name: Formula -> [Brand] [Gender] [Color] [Pattern] [Material] [Category]. Include top keywords.
    2. Style Note: Suggest where to wear (Occasion) and what to pair with.
    3. Keywords/Tags: Output a comma-separated list. PRIORITIZE the "KEYWORDS TO USE" list provided above.

    RETURN ONLY THE RAW JSON OBJECT.
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
st.sidebar.title("☁️ Google Sheet DB")
saved_cats = get_all_categories()
if st.sidebar.button("🔄 Refresh"): st.rerun()

tab1, tab2, tab3, tab4 = st.tabs(["🛠️ 1. Config Rules", "📈 2. SEO Keywords", "🚀 3. Run Generator", "🗑️ 4. Manage"])

# ---------------- TAB 1: CONFIG SETUP ----------------
with tab1:
    st.header("Step 1: Define Category Rules")
    mode = st.radio("Mode", ["Create New", "Edit Existing"], horizontal=True)
    
    cat_name = ""
    headers = []
    master_options = {}
    default_mapping = []

    if mode == "Edit Existing":
        if saved_cats:
            edit_cat = st.selectbox("Select Category to Edit", saved_cats)
            if edit_cat:
                loaded_data = load_config_from_sheet(edit_cat)
                if loaded_data:
                    cat_name = loaded_data['category_name']
                    headers = loaded_data['headers']
                    master_options = loaded_data['master_data']
                    for col, rule in loaded_data['column_mapping'].items():
                        src_map = {"AI": "AI Generation", "INPUT": "Input Excel", "FIXED": "Fixed Value", "BLANK": "Leave Blank"}
                        default_mapping.append({"Column Name": col, "Source": src_map.get(rule['source'], "Leave Blank"), "Fixed Value (If Fixed)": rule['value']})
    else:
        cat_name = st.text_input("New Category Name (e.g. Women Sarees)")

    c1, c2 = st.columns(2)
    template_file = c1.file_uploader("Upload Template (.xlsx)", type=["xlsx"])
    master_file = c2.file_uploader("Upload Master Data (.xlsx)", type=["xlsx"])

    if template_file: headers = pd.read_excel(template_file).columns.tolist()
    if master_file: master_options = parse_master_data(master_file)

    if headers:
        st.divider()
        st.write("Map Columns:")
        if not default_mapping:
            for h in headers:
                default_mapping.append({"Column Name": h, "Source": "Leave Blank", "Fixed Value (If Fixed)": ""})

        edited_df = st.data_editor(pd.DataFrame(default_mapping), column_config={"Source": st.column_config.SelectboxColumn("Source", options=["Input Excel", "AI Generation", "Fixed Value", "Leave Blank"])}, hide_index=True, use_container_width=True)
        
        if st.button("Save Config Rules"):
            final_mapping = {}
            for index, row in edited_df.iterrows():
                src_code = "AI" if row['Source'] == "AI Generation" else "INPUT" if row['Source'] == "Input Excel" else "FIXED" if row['Source'] == "Fixed Value" else "BLANK"
                final_mapping[row['Column Name']] = {"source": src_code, "value": row['Fixed Value (If Fixed)']}
            
            if save_config_to_sheet(cat_name, {"category_name": cat_name, "headers": headers, "master_data": master_options, "column_mapping": final_mapping}):
                st.success(f"✅ Rules for '{cat_name}' saved!"); time.sleep(1); st.rerun()

# ---------------- TAB 2: SEO KEYWORDS (NEW) ----------------
with tab2:
    st.header("Step 2: Upload Keywords (Ads Report)")
    st.info("Upload your latest High-Performing Keywords here. The AI will prioritize these.")
    
    if not saved_cats:
        st.warning("Create a category in Tab 1 first.")
    else:
        seo_cat = st.selectbox("Select Category for Keywords", saved_cats, key="seo_cat")
        
        # Display current keywords
        current_kw = get_keywords_for_category(seo_cat)
        if current_kw:
            st.caption("✅ Current Active Keywords in Database:")
            st.info(current_kw[:300] + "..." if len(current_kw) > 300 else current_kw)
        else:
            st.warning("No keywords uploaded yet.")

        kw_file = st.file_uploader("Upload Keywords File (.xlsx / .csv)", type=["xlsx", "csv"])
        
        if kw_file and st.button("Update Keywords"):
            try:
                if kw_file.name.endswith('.csv'): df_kw = pd.read_csv(kw_file)
                else: df_kw = pd.read_excel(kw_file)
                
                # Assume first column contains the keywords
                kw_list = df_kw.iloc[:, 0].dropna().astype(str).tolist()
                
                if save_keywords_to_sheet(seo_cat, kw_list):
                    st.success(f"✅ Updated {len(kw_list)} keywords for {seo_cat}!")
                    time.sleep(1); st.rerun()
            except Exception as e:
                st.error(f"Error reading file: {e}")

# ---------------- TAB 3: RUNNER ----------------
with tab3:
    st.header("Step 3: Generate Listings")
    
    if not saved_cats: st.warning("No configs found."); st.stop()
    
    run_cat = st.selectbox("Select Category", saved_cats, key="run_cat")
    
    # Check status
    config = load_config_from_sheet(run_cat)
    active_kws = get_keywords_for_category(run_cat)
    
    if active_kws: st.caption(f"✨ Using Custom Keywords for optimization")
    else: st.caption("⚠️ No custom keywords found. Using generic AI SEO.")

    input_data_file = st.file_uploader("Upload Input Data (.xlsx)", type=["xlsx"], key="run_input")
    
    if input_data_file and st.button("▶️ Start Processing"):
        df_input = pd.read_excel(input_data_file)
        if "Front Image" not in df_input.columns: st.error("Missing 'Front Image' column"); st.stop()
            
        progress_bar = st.progress(0); status = st.empty()
        final_rows = []
        image_cache = {}
        mapping = config['column_mapping']
        
        for index, row in df_input.iterrows():
            sku = row.get('vendorSkuCode', f'Row {index}')
            status.text(f"Processing: {sku}")
            progress_bar.progress((index + 1) / len(df_input))
            
            img_link = str(row.get('Front Image', '')).strip()
            ai_data = None
            needs_ai = any(m['source'] == 'AI' for m in mapping.values())
            
            if needs_ai:
                if img_link in image_cache: ai_data = image_cache[img_link]
                else:
                    hints = f"Color: {row.get('Brand Colour', '')}, Fabric: {row.get('Fabric', '')}"
                    # Pass the active keywords here
                    ai_data, _ = analyze_image_configured(client, img_link, hints, active_kws, config)
                    if ai_data: image_cache[img_link] = ai_data
            
            new_row = {}
            for col in config['headers']:
                rule = mapping.get(col, {'source': 'BLANK'})
                if rule['source'] == 'INPUT':
                    val = ""
                    if col in df_input.columns: val = row[col]
                    else:
                        for inp_col in df_input.columns:
                            if inp_col.lower() in col.lower(): val = row[inp_col]; break
                    new_row[col] = val
                elif rule['source'] == 'FIXED': new_row[col] = rule['value']
                elif rule['source'] == 'AI' and ai_data:
                    found = False
                    if col in ai_data: new_row[col] = ai_data[col]; found = True
                    else:
                        for k, v in ai_data.items():
                            if k.lower().replace(" ","") in col.lower().replace(" ",""): new_row[col] = v; found = True; break
                    if not found: new_row[col] = ""
                else: new_row[col] = ""
            final_rows.append(new_row)
        
        out_df = pd.DataFrame(final_rows)
        outfile = f"Result_{run_cat}_{int(time.time())}.xlsx"
        out_df.to_excel(outfile, index=False)
        st.success("✅ Done!")
        with open(outfile, "rb") as f: st.download_button("📥 Download Excel", f, file_name=outfile)

# ---------------- TAB 4: MANAGE ----------------
with tab4:
    st.header("Manage Configs")
    to_del = st.selectbox("Delete Category", [""] + saved_cats)
    if to_del and st.button("Delete"):
        delete_config_from_sheet(to_del)
        st.success("Deleted"); time.sleep(1); st.rerun()
