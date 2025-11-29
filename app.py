import streamlit as st
import pandas as pd
import json
import base64
import requests
from openai import OpenAI
import time

st.set_page_config(page_title="Myntra Dynamic Agent", layout="wide")

# --- SECURE KEY ---
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
except:
    st.error("❌ API Key not found in Secrets!")
    st.stop()

# --- LOAD DEFAULT KNOWLEDGE BASE ---
try:
    with open('categories.json', 'r') as f:
        CATEGORY_RULES = json.load(f)
except:
    st.error("❌ 'categories.json' not found.")
    st.stop()

# ================= SIDEBAR CONFIGURATION =================
st.sidebar.header("⚙️ Configuration")

# 1. Select Base Knowledge
selected_category = st.sidebar.selectbox("1. AI Knowledge Base", list(CATEGORY_RULES.keys()))
base_rules = CATEGORY_RULES[selected_category]

# 2. Dynamic Rule Editing (Edit Fabrics/Colors on the fly)
st.sidebar.markdown("### 2. Customize Valid Options")
# We let you edit the fabric list directly in the app
default_fabrics = ", ".join(base_rules['valid_options'].get('Fabric', []))
custom_fabrics = st.sidebar.text_area("Valid Fabrics (comma separated)", value=default_fabrics, height=100)

# Update the rules with your custom list
current_rules = base_rules.copy()
current_rules['valid_options']['Fabric'] = [x.strip() for x in custom_fabrics.split(',')]

# ================= MAIN APP =================
st.title("🛍️ Myntra Dynamic Listing Agent")
st.markdown("Automate **Any Category** by uploading the target template.")

col1, col2 = st.columns(2)

with col1:
    st.info("Step 1: Upload the empty Myntra Template (Defines Columns & Order)")
    template_file = st.file_uploader("Upload Target Template (.xlsx)", type=["xlsx"])

with col2:
    st.info("Step 2: Upload your Raw Product Data")
    input_file = st.file_uploader("Upload Input Data (.xlsx)", type=["xlsx"])

# --- DETERMINE HEADERS ---
target_headers = []

if template_file:
    # If user uploads a template, WE USE THAT exactly.
    temp_df = pd.read_excel(template_file)
    target_headers = temp_df.columns.tolist()
    st.success(f"✅ Template Loaded! Detected {len(target_headers)} columns (starting with '{target_headers[0]}').")
else:
    # Fallback to JSON headers if no template provided
    target_headers = current_rules['headers']
    st.warning("⚠️ No Template uploaded. Using default headers from JSON.")

# --- MAPPING DICTIONARY (Smart Auto-Map) ---
COLUMN_MAPPING = {
    "Manufacturer Name and Address with Pincode": ["Manufacturer Name", "Manufacturer", "Mfg Name"],
    "Packer Name and Address with Pincode": ["Packer Name", "Packer"],
    "Importer Name and Address with Pincode": ["Importer Name", "Importer"],
    "Brand Colour (Remarks)": ["Brand Colour", "Color", "Colour"],
    "vendorSkuCode": ["SKU", "Style Code", "Design No"],
    "Front Image": ["Image Link", "Image", "Front Image Link"]
}

# --- FUNCTIONS ---
def encode_image_from_url(url):
    try:
        if pd.isna(url) or str(url).strip() == "": return None, "Empty URL"
        if "dropbox.com" in url:
            url = url.replace("?dl=0", "").replace("&dl=0", "") + ("&dl=1" if "?" in url else "?dl=1")
        response = requests.get(url, timeout=10)
        return (base64.b64encode(response.content).decode('utf-8'), None) if response.status_code == 200 else (None, "Download Error")
    except Exception as e:
        return None, str(e)

def analyze_image(client, image_url, user_inputs, rules):
    base64_image, error = encode_image_from_url(image_url)
    if error: return None, error

    prompt = f"""
    {rules.get('system_prompt', '')}
    
    USER INPUTS: {user_inputs}
    
    VALID OPTIONS (Strictly enforce these): 
    {json.dumps(rules['valid_options'], indent=2)}
    
    RETURN RAW JSON.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a JSON-only assistant. Fill ALL fields."},
                {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
            ],
            response_format={"type": "json_object"},
            max_tokens=1500
        )
        return json.loads(response.choices[0].message.content), None
    except Exception as e:
        return None, str(e)

# --- EXECUTION ---
if input_file and st.button("🚀 Start Generation"):
    if not api_key: st.stop()
    
    df = pd.read_excel(input_file)
    
    # 1. SMART COLUMN RENAMING
    for target_col, possible_names in COLUMN_MAPPING.items():
        for name in possible_names:
            if name in df.columns:
                df = df.rename(columns={name: target_col})
                break
    
    if "Front Image" not in df.columns:
        st.error(f"❌ Error: Could not find 'Front Image' column.")
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()
    final_rows = []
    image_cache = {}
    
    for index, row in df.iterrows():
        sku = row.get('vendorSkuCode', f'Row {index}')
        status_text.text(f"Processing: {sku}")
        progress_bar.progress((index + 1) / len(df))
        
        img_link = str(row.get('Front Image', '')).strip()
        
        # Cache Check
        if img_link in image_cache:
            ai_data = image_cache[img_link]
        else:
            hints = f"Color: {row.get('Brand Colour (Remarks)', '')}, Fabric: {row.get('Fabric', '')}"
            ai_data, _ = analyze_image(client, img_link, hints, current_rules)
            if ai_data: image_cache[img_link] = ai_data

        # --- BUILDING THE ROW (Dynamic Mapping) ---
        # 1. Start with Empty Dictionary based on TARGET HEADERS (From Template)
        new_row = {header: "" for header in target_headers}
        
        # 2. Fill from Input Excel
        for col in df.columns:
            if col in new_row:
                new_row[col] = row[col]
        
        # 3. Fill from AI
        if ai_data:
            for k, v in ai_data.items():
                if k in new_row:
                    new_row[k] = v
        
        # 4. Defaults
        if 'defaults' in current_rules:
             for k, v in current_rules['defaults'].items():
                if k in new_row: new_row[k] = v

        new_row['Status'] = "Success" if ai_data else "Failed"
        final_rows.append(new_row)

    # --- FINAL OUTPUT ---
    output_df = pd.DataFrame(final_rows)
    
    # FORCE ORDER based on Template
    ordered_cols = [h for h in target_headers if h in output_df.columns]
    output_df = output_df[ordered_cols + ['Status']]
    
    outfile = f"Myntra_Result_{int(time.time())}.xlsx"
    output_df.to_excel(outfile, index=False)
    
    st.success("✅ Done!")
    with open(outfile, "rb") as f:
        st.download_button("📥 Download Final Result", f, file_name=outfile)
