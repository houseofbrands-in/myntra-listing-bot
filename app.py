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
    # We construct the creds dictionary from Streamlit Secrets
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    gc = gspread.authorize(creds)
    
    # CONNECT TO SHEET
    SHEET_NAME = "Agency_OS_Database" # Make sure your Google Sheet has this exact name
    worksheet = gc.open(SHEET_NAME).worksheet("Configs")
    
except Exception as e:
    st.error(f"❌ Connection Error: {str(e)}")
    st.info("Check Secrets and ensure Google Sheet is shared with the Service Account Email.")
    st.stop()

# ================= DATABASE FUNCTIONS (Google Sheets) =================
def get_all_categories():
    try:
        # Get all values in Column A (Category Names)
        # Skip header (row 1)
        cats = worksheet.col_values(1)[1:] 
        return [c for c in cats if c] # Remove empty
    except:
        return []

def save_config_to_sheet(name, data):
    try:
        json_str = json.dumps(data)
        
        # Check if category exists
        cell = None
        try:
            cell = worksheet.find(name)
        except:
            pass # Not found
            
        if cell:
            # Update existing row (Column B is index 2)
            worksheet.update_cell(cell.row, 2, json_str)
        else:
            # Append new row
            worksheet.append_row([name, json_str])
            
        return True
    except Exception as e:
        st.error(f"Save Failed: {e}")
        return False

def load_config_from_sheet(name):
    try:
        cell = worksheet.find(name)
        json_str = worksheet.cell(cell.row, 2).value
        return json.loads(json_str)
    except:
        return None

def delete_config_from_sheet(name):
    try:
        cell = worksheet.find(name)
        worksheet.delete_rows(cell.row)
        return True
    except:
        return False

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

def analyze_image_configured(client, image_url, user_inputs, config):
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

    prompt = f"""
    You are a Myntra Cataloging Expert.
    CATEGORY: {config['category_name']}
    USER HINTS: {user_inputs}
    
    TASK: Analyze the image and fill these specific attributes: {ai_target_headers}
    
    STRICT VALIDATION RULES (Choose ONLY from here):
    {json.dumps(relevant_options, indent=2)}
    
    MANDATORY TEXT FIELDS:
    - vendorArticleName (Catchy title)
    - product_description (Marketing copy)
    - productDetails (Bullet points)
    - styleNote (Styling tip)
    - materialCareDescription (Care info)
    - sizeAndFitDescription (Model info)
    
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

# ================= APP UI =================
st.sidebar.title("☁️ Google Sheet DB")
saved_cats = get_all_categories()

if st.sidebar.button("🔄 Refresh Categories"):
    st.rerun()

st.sidebar.write(f"Synced Categories: {len(saved_cats)}")
if saved_cats:
    st.sidebar.selectbox("View Available:", saved_cats, disabled=True)

tab1, tab2, tab3 = st.tabs(["🛠️ Create/Edit Config", "🚀 Run Generator", "🗑️ Manage"])

# ---------------- TAB 1: SETUP ----------------
with tab1:
    st.header("Step 1: Define or Edit a Category")
    mode = st.radio("Mode", ["Create New", "Edit Existing"], horizontal=True)
    
    cat_name = ""
    headers = []
    master_options = {}
    default_mapping = []

    if mode == "Edit Existing":
        if not saved_cats:
            st.warning("No categories found in Google Sheet.")
        else:
            edit_cat = st.selectbox("Select Category to Edit", saved_cats)
            if edit_cat:
                loaded_data = load_config_from_sheet(edit_cat)
                if loaded_data:
                    cat_name = loaded_data['category_name']
                    headers = loaded_data['headers']
                    master_options = loaded_data['master_data']
                    for col, rule in loaded_data['column_mapping'].items():
                        src_map = {"AI": "AI Generation", "INPUT": "Input Excel", "FIXED": "Fixed Value", "BLANK": "Leave Blank"}
                        default_mapping.append({
                            "Column Name": col,
                            "Source": src_map.get(rule['source'], "Leave Blank"),
                            "Fixed Value (If Fixed)": rule['value']
                        })
    else:
        cat_name = st.text_input("New Category Name (e.g. Women Sarees)")

    col_a, col_b = st.columns(2)
    with col_a:
        template_file = st.file_uploader("1. Upload Myntra Template (.xlsx)", type=["xlsx"])
    with col_b:
        master_file = st.file_uploader("2. Upload Master Data (.xlsx)", type=["xlsx"])

    if template_file:
        headers = pd.read_excel(template_file).columns.tolist()
        default_mapping = [] 
    if master_file:
        master_options = parse_master_data(master_file)

    if headers:
        st.divider()
        st.subheader(f"Mapping Columns for: {cat_name}")
        
        if not default_mapping:
            for h in headers:
                default_source = "Leave Blank"
                default_fixed = ""
                h_lower = h.lower()
                if "image" in h_lower or "sku" in h_lower or "mrp" in h_lower or "brand" in h_lower: default_source = "Input Excel"
                elif h in master_options or "fabric" in h_lower or "neck" in h_lower or "pattern" in h_lower: default_source = "AI Generation"
                elif "year" in h_lower: default_source = "Fixed Value"; default_fixed = "2026"
                elif "fashiontype" in h_lower: default_source = "Fixed Value"; default_fixed = "Fashion"
                default_mapping.append({"Column Name": h, "Source": default_source, "Fixed Value (If Fixed)": default_fixed})

        edited_df = st.data_editor(
            pd.DataFrame(default_mapping),
            column_config={
                "Source": st.column_config.SelectboxColumn("Source", options=["Input Excel", "AI Generation", "Fixed Value", "Leave Blank"], required=True)
            },
            hide_index=True,
            use_container_width=True,
            height=500
        )
        
        if st.button("☁️ Save to Google Sheet"):
            if not cat_name: st.error("Category Name required"); st.stop()
            
            final_mapping = {}
            for index, row in edited_df.iterrows():
                src_code = "AI" if row['Source'] == "AI Generation" else "INPUT" if row['Source'] == "Input Excel" else "FIXED" if row['Source'] == "Fixed Value" else "BLANK"
                final_mapping[row['Column Name']] = {"source": src_code, "value": row['Fixed Value (If Fixed)']}
            
            config_payload = {
                "category_name": cat_name,
                "headers": headers,
                "master_data": master_options,
                "column_mapping": final_mapping
            }
            
            if save_config_to_sheet(cat_name, config_payload):
                st.success(f"✅ '{cat_name}' synced to Cloud! Your team can now see it.")
                time.sleep(2)
                st.rerun()

# ---------------- TAB 2: RUNNER ----------------
with tab2:
    st.header("🚀 Generate Listings (Cloud Mode)")
    
    if not saved_cats:
        st.warning("No configs found in Sheet.")
    else:
        selected_config_name = st.selectbox("Select Category to Run", saved_cats)
        input_data_file = st.file_uploader("Upload Input Data (.xlsx)", type=["xlsx"])
        
        if input_data_file and st.button("▶️ Start Processing"):
            config = load_config_from_sheet(selected_config_name)
            df_input = pd.read_excel(input_data_file)
            
            if "Front Image" not in df_input.columns:
                st.error("❌ Input Excel must have a column named 'Front Image'.")
                st.stop()
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            final_rows = []
            image_cache = {}
            
            target_headers = config['headers']
            mapping = config['column_mapping']
            
            for index, row in df_input.iterrows():
                sku = row.get('vendorSkuCode', f'Row {index}')
                status_text.text(f"Processing: {sku}")
                progress_bar.progress((index + 1) / len(df_input))
                
                img_link = str(row.get('Front Image', '')).strip()
                
                ai_data = None
                needs_ai = any(m['source'] == 'AI' for m in mapping.values())
                
                if needs_ai:
                    if img_link in image_cache:
                        ai_data = image_cache[img_link]
                    else:
                        hints = f"Color: {row.get('Brand Colour', '')}, Fabric: {row.get('Fabric', '')}"
                        ai_data, _ = analyze_image_configured(client, img_link, hints, config)
                        if ai_data: image_cache[img_link] = ai_data
                
                new_row = {}
                for col in target_headers:
                    rule = mapping.get(col, {'source': 'BLANK'})
                    if rule['source'] == 'INPUT':
                        val = ""
                        if col in df_input.columns: val = row[col]
                        else:
                            for inp_col in df_input.columns:
                                if inp_col.lower() in col.lower() or col.lower() in inp_col.lower():
                                    val = row[inp_col]; break
                        new_row[col] = val
                    elif rule['source'] == 'FIXED':
                        new_row[col] = rule['value']
                    elif rule['source'] == 'AI' and ai_data:
                        found = False
                        for k, v in ai_data.items():
                            if k.lower().replace(" ","") == col.lower().replace(" ",""):
                                new_row[col] = v; found = True; break
                        if not found: new_row[col] = ""
                    else:
                        new_row[col] = ""

                new_row['Status'] = "Success"
                final_rows.append(new_row)
            
            out_df = pd.DataFrame(final_rows)
            out_df = out_df[target_headers + ['Status']]
            outfile = f"Result_{selected_config_name}_{int(time.time())}.xlsx"
            out_df.to_excel(outfile, index=False)
            
            st.success("✅ Done!")
            with open(outfile, "rb") as f:
                st.download_button("📥 Download Final Excel", f, file_name=outfile)

# ---------------- TAB 3: MANAGE ----------------
with tab3:
    st.header("🗑️ Manage Cloud Configs")
    to_delete = st.selectbox("Select Config to Delete", [""] + saved_cats)
    if to_delete and st.button(f"Delete '{to_delete}' from Cloud"):
        if delete_config_from_sheet(to_delete):
            st.success(f"Deleted {to_delete}")
            time.sleep(1)
            st.rerun()
