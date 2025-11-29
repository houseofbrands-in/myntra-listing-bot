import streamlit as st
import pandas as pd
import json
import base64
import requests
from openai import OpenAI
import time

# ================= SETUP =================
st.set_page_config(page_title="Myntra Listing Bot", layout="wide")

st.title("🛍️ Automated Myntra Listing Engine")
st.markdown("Upload your Excel, pick a category, and let AI write the catalog.")

# 1. API KEY INPUT (Secure)
api_key = st.text_input("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", type="password")

# 2. LOAD CATEGORY RULES
try:
    with open('categories.json', 'r') as f:
        CATEGORY_RULES = json.load(f)
except FileNotFoundError:
    st.error("⚠️ 'categories.json' file not found!")
    st.stop()

# 3. SELECT CATEGORY
selected_category = st.selectbox("Select Category", list(CATEGORY_RULES.keys()))
current_rules = CATEGORY_RULES[selected_category]

# Display valid options for the user to see
with st.expander(f"View Rules for {selected_category}"):
    st.json(current_rules['valid_options'])

# ================= HELPER FUNCTIONS =================
def encode_image_from_url(url):
    try:
        if pd.isna(url) or str(url).strip() == "": return None
        if "dropbox.com" in url:
            url = url.replace("?dl=0", "").replace("&dl=0", "") + ("&dl=1" if "?" in url else "?dl=1")
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8')
    except:
        return None
    return None

def analyze_image(client, image_url, user_inputs, rules):
    base64_image = encode_image_from_url(image_url)
    if not base64_image: return None

    # Dynamic Prompt based on selected category
    valid_opts_str = json.dumps(rules['valid_options'], indent=2)
    
    prompt = f"""
    You are a Catalog AI. 
    Category: {selected_category}
    User Hints: {user_inputs}
    
    Task: Analyze the image and fill the attributes.
    STRICT RULE: You must choose values ONLY from this list:
    {valid_opts_str}
    
    Also generate:
    - vendorArticleName (Catchy title)
    - product_description (100 words marketing copy)
    - productDetails (Bullet points)

    Return ONLY raw JSON.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a JSON-only assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ]}
            ],
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return None

# ================= MAIN APP LOGIC =================
uploaded_file = st.file_uploader("Upload Input Excel", type=["xlsx"])

if uploaded_file and api_key:
    if st.button("🚀 Start Generation"):
        client = OpenAI(api_key=api_key)
        df = pd.read_excel(uploaded_file)
        
        # Progress Bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        final_rows = []
        image_cache = {}
        
        total = len(df)
        
        for index, row in df.iterrows():
            # Update UI
            status_text.text(f"Processing Row {index + 1}/{total} - SKU: {row.get('vendorSkuCode', 'Unknown')}")
            progress_bar.progress((index + 1) / total)
            
            img_link = str(row.get('Front Image', '')).strip()
            
            # --- CACHE LOGIC ---
            if img_link in image_cache:
                ai_data = image_cache[img_link]
            else:
                # Gather hints from user columns
                hints = f"Color: {row.get('Brand Colour', '')}, Fabric: {row.get('Fabric', '')}"
                ai_data = analyze_image(client, img_link, hints, current_rules)
                if ai_data: image_cache[img_link] = ai_data
            
            if not ai_data: continue

            # --- MAPPING ---
            # Start with empty row based on the Category's headers
            new_row = {col: "" for col in current_rules['headers']}
            
            # Fill Standard Data from Input
            for col in df.columns:
                if col in new_row:
                    new_row[col] = row[col]
            
            # Fill AI Data
            for key, value in ai_data.items():
                if key in new_row:
                    new_row[key] = value
            
            final_rows.append(new_row)
            
        # --- FINISH ---
        output_df = pd.DataFrame(final_rows)
        
        st.success("✅ Processing Complete!")
        
        # Create Download Button
        csv = output_df.to_csv(index=False).encode('utf-8')
        excel_file = f"Myntra_{selected_category}_{int(time.time())}.xlsx"
        output_df.to_excel(excel_file, index=False)
        
        with open(excel_file, "rb") as f:
            st.download_button(
                label="📥 Download Excel Result",
                data=f,
                file_name=excel_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )