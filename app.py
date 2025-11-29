import streamlit as st
import pandas as pd
import json
import base64
import requests
from openai import OpenAI
import time

# ================= SETUP =================
st.set_page_config(page_title="Myntra Listing Bot", layout="wide")

st.title("🛍️ Automated Myntra Listing Engine (Debug Mode)")
st.markdown("Upload your Excel. If it fails, check the 'Status Log' below.")

# 1. API KEY INPUT
api_key = st.text_input("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", type="password")

# 2. LOAD CATEGORY RULES
try:
    with open('categories.json', 'r') as f:
        CATEGORY_RULES = json.load(f)
except FileNotFoundError:
    st.error("⚠️ 'categories.json' file not found! Please upload it to GitHub.")
    st.stop()

# 3. SELECT CATEGORY
selected_category = st.selectbox("Select Category", list(CATEGORY_RULES.keys()))
current_rules = CATEGORY_RULES[selected_category]

# Display valid options
with st.expander(f"View Rules for {selected_category}"):
    st.json(current_rules['valid_options'])

# ================= HELPER FUNCTIONS =================
def encode_image_from_url(url):
    try:
        if pd.isna(url) or str(url).strip() == "": 
            return None, "Empty URL"
        
        # Dropbox fix
        if "dropbox.com" in url:
            url = url.replace("?dl=0", "").replace("&dl=0", "") + ("&dl=1" if "?" in url else "?dl=1")
            
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8'), None
        else:
            return None, f"Download Error (Status: {response.status_code})"
    except Exception as e:
        return None, f"Download Exception: {str(e)}"

def analyze_image(client, image_url, user_inputs, rules):
    base64_image, error = encode_image_from_url(image_url)
    if error: return None, error

    valid_opts_str = json.dumps(rules['valid_options'], indent=2)
    
    prompt = f"""
    You are a Catalog AI. Category: {selected_category}. User Hints: {user_inputs}
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
        return json.loads(response.choices[0].message.content), None
    except Exception as e:
        return None, f"OpenAI API Error: {str(e)}"

# ================= MAIN APP LOGIC =================
uploaded_file = st.file_uploader("Upload Input Excel", type=["xlsx"])

if uploaded_file and api_key:
    if st.button("🚀 Start Generation"):
        try:
            client = OpenAI(api_key=api_key)
            df = pd.read_excel(uploaded_file)
            
            # CHECK COLUMN NAMES
            st.write("### Debug Info:")
            st.write(f"Columns found in Excel: {list(df.columns)}")
            
            if "Front Image" not in df.columns:
                st.error("❌ Error: Column 'Front Image' not found! Please rename your image link column to 'Front Image'.")
                st.stop()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            final_rows = []
            image_cache = {}
            total = len(df)
            
            for index, row in df.iterrows():
                sku = row.get('vendorSkuCode', f'Row {index+1}')
                status_text.text(f"Processing: {sku}")
                progress_bar.progress((index + 1) / total)
                
                img_link = str(row.get('Front Image', '')).strip()
                error_msg = ""
                ai_data = None

                # --- CACHE LOGIC ---
                if img_link in image_cache:
                    ai_data = image_cache[img_link]
                else:
                    hints = f"Color: {row.get('Brand Colour', '')}, Fabric: {row.get('Fabric', '')}"
                    ai_data, error_msg = analyze_image(client, img_link, hints, current_rules)
                    if ai_data: 
                        image_cache[img_link] = ai_data
                
                # --- MAPPING ---
                # Prepare row with defaults
                new_row = {col: "" for col in current_rules['headers']}
                
                # Fill Input Data
                for col in df.columns:
                    if col in new_row:
                        new_row[col] = row[col]
                
                # Fill AI Data or Error
                if ai_data:
                    for key, value in ai_data.items():
                        if key in new_row:
                            new_row[key] = value
                    new_row['Status'] = "Success"
                else:
                    new_row['Status'] = f"Failed: {error_msg}"
                
                final_rows.append(new_row)
                
            # --- FINISH ---
            output_df = pd.DataFrame(final_rows)
            
            st.success("✅ Processing Complete!")
            
            # Show preview of failures if any
            failed_rows = output_df[output_df['Status'].str.contains("Failed")]
            if not failed_rows.empty:
                st.warning(f"⚠️ {len(failed_rows)} rows failed. Check the 'Status' column in the download.")
                st.dataframe(failed_rows[['vendorSkuCode', 'Status']])

            # Download
            excel_file = f"Myntra_Result_{int(time.time())}.xlsx"
            output_df.to_excel(excel_file, index=False)
            
            with open(excel_file, "rb") as f:
                st.download_button(
                    label="📥 Download Result (With Error Logs)",
                    data=f,
                    file_name=excel_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
        except Exception as e:
            st.error(f"Critical Error: {str(e)}")

