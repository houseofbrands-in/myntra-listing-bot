import streamlit as st
import pandas as pd
import json
import base64
import requests
from openai import OpenAI
import time

st.set_page_config(page_title="Myntra Agency OS", layout="wide")

# --- SECURE KEY ---
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
except:
    st.error("❌ API Key not found in Secrets!")
    st.stop()

# ================= HELPER FUNCTIONS =================
def parse_master_data(file):
    """Extracts valid options from Master Data Sheet"""
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

    # Filter master data to only show options for columns marked as "AI"
    relevant_options = {}
    ai_target_headers = []
    
    for col, settings in config['column_mapping'].items():
        if settings['source'] == 'AI':
            ai_target_headers.append(col)
            # Find matching master data options
            # We try to match column names loosely
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

# ================= TABS LOGIC =================
tab1, tab2 = st.tabs(["🛠️ Create Category Config", "🚀 Run Generator"])

# ---------------- TAB 1: SETUP ----------------
with tab1:
    st.header("Step 1: Define a New Category")
    cat_name = st.text_input("Category Name (e.g. Women Sarees)", "Women Sarees")
    
    col_a, col_b = st.columns(2)
    with col_a:
        template_file = st.file_uploader("1. Upload Myntra Blank Template (.xlsx)", type=["xlsx"])
    with col_b:
        master_file = st.file_uploader("2. Upload Master Data / Dropdowns (.xlsx)", type=["xlsx"])

    if template_file and master_file:
        # Load Data
        df_temp = pd.read_excel(template_file)
        headers = df_temp.columns.tolist()
        master_options = parse_master_data(master_file)
        
        st.divider()
        st.subheader("Step 2: Map Your Columns")
        st.info("Define where data for each column should come from.")
        
        # Mapping Interface
        mapping_config = {}
        
        # We create a dataframe editor for easier mapping
        mapping_data = []
        for h in headers:
            # Smart Default Guessing
            default_source = "Leave Blank"
            default_fixed = ""
            
            h_lower = h.lower()
            if "image" in h_lower or "sku" in h_lower or "mrp" in h_lower or "brand" in h_lower:
                default_source = "Input Excel"
            elif h in master_options or "fabric" in h_lower or "neck" in h_lower or "pattern" in h_lower or "description" in h_lower:
                default_source = "AI Generation"
            elif "year" in h_lower:
                default_source = "Fixed Value"
                default_fixed = "2026"
            elif "fashiontype" in h_lower:
                default_source = "Fixed Value"
                default_fixed = "Fashion"
                
            mapping_data.append({
                "Column Name": h,
                "Source": default_source,
                "Fixed Value (If Fixed)": default_fixed
            })
            
        edited_df = st.data_editor(
            pd.DataFrame(mapping_data),
            column_config={
                "Source": st.column_config.SelectboxColumn(
                    "Source",
                    options=["Input Excel", "AI Generation", "Fixed Value", "Leave Blank"],
                    required=True
                )
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )
        
        if st.button("💾 Save & Download Configuration"):
            # Convert UI selection back to Config JSON
            final_mapping = {}
            for index, row in edited_df.iterrows():
                final_mapping[row['Column Name']] = {
                    "source": "AI" if row['Source'] == "AI Generation" else 
                              "INPUT" if row['Source'] == "Input Excel" else
                              "FIXED" if row['Source'] == "Fixed Value" else "BLANK",
                    "value": row['Fixed Value (If Fixed)']
                }
            
            config_payload = {
                "category_name": cat_name,
                "headers": headers,
                "master_data": master_options,
                "column_mapping": final_mapping
            }
            
            json_str = json.dumps(config_payload, indent=2)
            file_name = f"config_{cat_name.replace(' ', '_').lower()}.json"
            
            st.success("✅ Configuration Built! Download this file and use it in the 'Run Generator' tab.")
            st.download_button("📥 Download Config JSON", json_str, file_name=file_name)


# ---------------- TAB 2: RUNNER ----------------
with tab2:
    st.header("🚀 Generate Listings")
    
    c1, c2 = st.columns(2)
    with c1:
        config_file = st.file_uploader("1. Upload Config File (.json)", type=["json"])
    with c2:
        input_data_file = st.file_uploader("2. Upload Input Data (.xlsx)", type=["xlsx"])
        
    if config_file and input_data_file and st.button("▶️ Start Processing"):
        config = json.load(config_file)
        df_input = pd.read_excel(input_data_file)
        
        # Validate Input
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
            
            # 1. AI Processing (Only if needed)
            ai_data = None
            needs_ai = any(m['source'] == 'AI' for m in mapping.values())
            
            if needs_ai:
                if img_link in image_cache:
                    ai_data = image_cache[img_link]
                else:
                    hints = f"Color: {row.get('Brand Colour', '')}, Fabric: {row.get('Fabric', '')}"
                    ai_data, _ = analyze_image_configured(client, img_link, hints, config)
                    if ai_data: image_cache[img_link] = ai_data
            
            # 2. Build Row based on Mapping
            new_row = {}
            for col in target_headers:
                rule = mapping.get(col, {'source': 'BLANK'})
                
                if rule['source'] == 'INPUT':
                    # Try to find matching column in input
                    # Logic: Try exact match, then loose match
                    val = ""
                    if col in df_input.columns:
                        val = row[col]
                    else:
                        # Fallback mapping
                        if "Manufacturer" in col and "Manufacturer Name" in df_input.columns: val = row["Manufacturer Name"]
                        elif "Packer" in col and "Packer Name" in df_input.columns: val = row["Packer Name"]
                        elif "Importer" in col and "Importer Name" in df_input.columns: val = row["Importer Name"]
                        elif "Brand Colour" in col and "Brand Colour" in df_input.columns: val = row["Brand Colour"]
                        elif "SKU" in col and "vendorSkuCode" in df_input.columns: val = row["vendorSkuCode"]
                    new_row[col] = val
                    
                elif rule['source'] == 'FIXED':
                    new_row[col] = rule['value']
                    
                elif rule['source'] == 'AI' and ai_data:
                    # Try to find key in AI result
                    # AI might return "Neck" or "neck_type", so we try fuzzy match
                    found = False
                    for k, v in ai_data.items():
                        if k.lower() == col.lower() or k.lower().replace(" ", "") == col.lower().replace(" ", ""):
                            new_row[col] = v
                            found = True
                            break
                    if not found: new_row[col] = ""
                    
                else:
                    new_row[col] = "" # Blank

            new_row['Status'] = "Success"
            final_rows.append(new_row)
            
        # Output
        out_df = pd.DataFrame(final_rows)
        # Enforce order
        out_df = out_df[target_headers + ['Status']]
        
        outfile = f"Result_{config['category_name']}_{int(time.time())}.xlsx"
        out_df.to_excel(outfile, index=False)
        
        st.success("✅ Done!")
        with open(outfile, "rb") as f:
            st.download_button("📥 Download Final Excel", f, file_name=outfile)
