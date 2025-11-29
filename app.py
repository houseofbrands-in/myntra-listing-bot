import streamlit as st
import pandas as pd
import json
import base64
import requests
from openai import OpenAI
import time

st.set_page_config(page_title="Myntra Listing Bot", layout="wide")

# --- SECURE KEY ---
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
except:
    st.error("❌ API Key not found in Secrets!")
    st.stop()

# --- LOAD RULES ---
try:
    with open('categories.json', 'r') as f:
        CATEGORY_RULES = json.load(f)
except:
    st.error("❌ 'categories.json' not found.")
    st.stop()

selected_category = st.selectbox("Select Category", list(CATEGORY_RULES.keys()))
current_rules = CATEGORY_RULES[selected_category]

# --- COLUMN MAPPING CONFIGURATION ---
# Format: "Myntra Header": ["Your Excel Header 1", "Your Excel Header 2"]
# This tells the script: "If you can't find 'Manufacturer...', look for 'Manufacturer Name' instead."
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
    VALID OPTIONS: {json.dumps(rules['valid_options'], indent=2)}
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

# --- MAIN APP ---
uploaded_file = st.file_uploader("Upload Input Excel", type=["xlsx"])

if uploaded_file and st.button("🚀 Start Generation"):
    df = pd.read_excel(uploaded_file)
    
    # 1. SMART COLUMN RENAMING
    # Renames your columns (e.g. "Manufacturer Name") to match Myntra ("Manufacturer Name and Address...")
    for target_col, possible_names in COLUMN_MAPPING.items():
        for name in possible_names:
            if name in df.columns:
                df = df.rename(columns={name: target_col})
                break
    
    if "Front Image" not in df.columns:
        st.error(f"❌ Error: Could not find 'Front Image' column. Found: {list(df.columns)}")
        st.stop()

    progress_bar = st.progress(0)
    final_rows = []
    image_cache = {}
    
    for index, row in df.iterrows():
        sku = row.get('vendorSkuCode', f'Row {index}')
        progress_bar.progress((index + 1) / len(df))
        
        img_link = str(row.get('Front Image', '')).strip()
        
        # Cache Check
        if img_link in image_cache:
            ai_data = image_cache[img_link]
        else:
            hints = f"Color: {row.get('Brand Colour (Remarks)', '')}, Fabric: {row.get('Fabric', '')}"
            ai_data, _ = analyze_image(client, img_link, hints, current_rules)
            if ai_data: image_cache[img_link] = ai_data

        # --- BUILDING THE ROW ---
        # 1. Start with Empty Dictionary based on Headers
        new_row = {header: "" for header in current_rules['headers']}
        
        # 2. Fill from Input Excel (Prioritize User Data)
        for col in df.columns:
            if col in new_row:
                new_row[col] = row[col]
        
        # 3. Fill from Category Defaults (e.g. ArticleType)
        if 'defaults' in current_rules:
            for k, v in current_rules['defaults'].items():
                if k in new_row: new_row[k] = v

        # 4. Fill from AI (If empty in Input)
        if ai_data:
            for k, v in ai_data.items():
                if k in new_row:
                    new_row[k] = v
        
        new_row['Status'] = "Success" if ai_data else "Failed"
        final_rows.append(new_row)

    # --- FINAL OUTPUT GENERATION ---
    output_df = pd.DataFrame(final_rows)
    
    # CRITICAL: Force the exact column order from JSON
    ordered_cols = [h for h in current_rules['headers'] if h in output_df.columns]
    output_df = output_df[ordered_cols + ['Status']] # Add status at the very end
    
    # Save & Download
    outfile = f"Myntra_Result_{int(time.time())}.xlsx"
    output_df.to_excel(outfile, index=False)
    with open(outfile, "rb") as f:
        st.download_button("📥 Download Final Result", f, file_name=outfile)
    
    st.success("✅ Done!")
