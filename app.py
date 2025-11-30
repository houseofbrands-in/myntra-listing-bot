import streamlit as st
import pandas as pd
import json
import base64
import requests
from io import BytesIO
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================

st.set_page_config(page_title="Agency OS", layout="wide")
st.title("Agency OS: Automated Cataloging")

# Authentication
api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai_key")
if api_key:
    client = OpenAI(api_key=api_key)
else:
    st.error("❌ API Key missing. Check secrets.toml")
    st.stop()

def connect_to_gsheets():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
        else:
            creds_dict = dict(st.secrets)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client_gs = gspread.authorize(creds)
        return client_gs.open("Agency_OS_Database").worksheet("Configs")
    except Exception as e:
        st.error(f"Database Error: {e}")
        return None

# ==========================================
# 2. HELPER: DOWNLOAD IMAGE (Dropbox Fixed)
# ==========================================
def download_image_from_url(url):
    if not isinstance(url, str):
        return None
    try:
        url = url.strip()
        # Dropbox Fix
        if "dropbox.com" in url:
            if "?dl=0" in url: url = url.replace("?dl=0", "?dl=1")
            elif "?dl=1" not in url: url += "?dl=1" if "?" in url else "?dl=1"

        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            return response.content
        return None
    except:
        return None

# ==========================================
# 3. AI LOGIC (CRITICAL FIX APPLIED)
# ==========================================
def analyze_image_configured(image_bytes, config_json, seo_keywords=""):
    # 1. Parse JSON if string
    if isinstance(config_json, str):
        try: config_json = json.loads(config_json)
        except: return {}

    # === BUG FIX: Normalize Data Structure ===
    # If config is a single dictionary, wrap it in a list
    if isinstance(config_json, dict):
        config_json = [config_json]
    # =========================================

    strict_fields = []
    creative_fields = []

    for field in config_json:
        # Safety check to ensure 'field' is actually a dict
        if not isinstance(field, dict): continue

        if field.get("Type") == "AI":
            col_name = field.get("Column")
            options = field.get("Options", [])
            
            # STRICT MODE
            if isinstance(options, list) and len(options) > 0:
                clean_opts = [str(opt).strip() for opt in options if pd.notna(opt)]
                options_str = ", ".join([f"'{opt}'" for opt in clean_opts])
                strict_fields.append(f"- **{col_name}**: Choose strictly from [{options_str}]")
            # CREATIVE MODE
            else:
                col_lower = col_name.lower()
                if "tag" in col_lower: guide = f"Generate SEO tags. Context: {seo_keywords}."
                elif "name" in col_lower: guide = f"Generate Product Name. Context: {seo_keywords}."
                else: guide = f"Write description. Context: {seo_keywords}."
                creative_fields.append(f"- **{col_name}**: {guide}")

    if not strict_fields and not creative_fields: return {}

    prompt_sections = []
    if strict_fields: prompt_sections.append("### STRICT FIELDS (Pick Exact Option)\n" + "\n".join(strict_fields))
    if creative_fields: prompt_sections.append("### CREATIVE FIELDS (Write Content)\n" + "\n".join(creative_fields))

    system_prompt = "You are a Cataloging AI. Output ONLY valid JSON."
    user_prompt = f"Analyze image.\n\n" + "\n\n".join(prompt_sections) + "\n\nReturn JSON object."
    
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=800,
            temperature=0.4, 
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"Error": str(e)}

# ==========================================
# 4. USER INTERFACE
# ==========================================
tab1, tab2, tab3 = st.tabs(["⚙️ Setup", "🚀 Run (Excel Input)", "✏️ Edit Configs"])

# --- TAB 1: SETUP ---
with tab1:
    st.header("1. Define Rules")
    c1, c2 = st.columns(2)
    template_file = c1.file_uploader("Upload Blank Template", type=["xlsx", "csv"])
    master_file = c2.file_uploader("Upload Master Data (Dropdowns)", type=["xlsx", "csv"])

    final_master_options = {}
    if master_file:
        df_master = pd.read_csv(master_file) if master_file.name.endswith('.csv') else pd.read_excel(master_file)
        st.caption("👇 Edit your dropdowns here if needed:")
        edited_df = st.data_editor(df_master, num_rows="dynamic")
        for col in edited_df.columns:
            final_master_options[col] = edited_df[col].dropna().unique().tolist()

    if template_file:
        st.divider()
        st.subheader("2. Map Columns")
        df_temp = pd.read_csv(template_file) if template_file.name.endswith('.csv') else pd.read_excel(template_file)
        
        config_builder = []
        with st.form("mapping_form"):
            for col in df_temp.columns:
                c_a, c_b = st.columns([2, 1])
                c_a.write(f"**{col}**")
                type_ = c_b.selectbox("Type", ["Fixed", "Input", "AI"], key=f"t_{col}", label_visibility="collapsed")
                
                field_data = {"Column": col, "Type": type_, "Options": []}
                if type_ == "AI" and col in final_master_options:
                    field_data["Options"] = final_master_options[col]
                config_builder.append(field_data)
            
            st.divider()
            name = st.text_input("Config Name (e.g. Myntra Kurta)")
            if st.form_submit_button("Save Config"):
                ws = connect_to_gsheets()
                ws.append_row([name, json.dumps(config_builder)])
                st.success("Saved!")

# --- TAB 2: RUN (EXCEL + URL SUPPORT) ---
with tab2:
    st.header("Generate Catalog from Excel")
    ws = connect_to_gsheets()
    all_data = ws.get_all_values() if ws else []
    config_names = [r[0] for r in all_data]
    
    selected_conf = st.selectbox("1. Select Category", config_names)
    input_excel = st.file_uploader("2. Upload Input Excel", type=['xlsx'])
    seo_keywords = st.text_area("3. SEO Keywords (Optional)", placeholder="Summer, Cotton, etc.")

    if st.button("Generate") and input_excel:
        json_str = next((r[1] for r in all_data if r[0] == selected_conf), "[]")
        df_input = pd.read_excel(input_excel)
        
        # FIND IMAGE COLUMN (flexible matching)
        img_col = next((c for c in df_input.columns if "link" in c.lower() or "url" in c.lower() or "image" in c.lower()), None)
        
        if not img_col:
            st.error("❌ Could not find a column named 'Image Link', 'Image URL' or 'Image' in your Excel.")
        else:
            st.success(f"✅ Found Image Column: '{img_col}'. Starting processing...")
            progress = st.progress(0)
            
            for index, row in df_input.iterrows():
                url = str(row[img_col]).strip()
                
                # Download Image (Handles Dropbox)
                img_bytes = download_image_from_url(url)
                
                if img_bytes:
                    # Run AI
                    ai_result = analyze_image_configured(img_bytes, json_str, seo_keywords)
                    
                    # Update Row
                    for key, value in ai_result.items():
                        if key in df_input.columns:
                            df_input.at[index, key] = value
                else:
                    st.warning(f"⚠️ Row {index+1}: Could not download image. Check if link is valid.")
                
                progress.progress((index + 1) / len(df_input))
            
            # Export
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_input.to_excel(writer, index=False)
            output.seek(0)
            st.success("Processing Complete!")
            st.download_button("Download Output File", data=output, file_name="Generated_Catalog.xlsx")

# --- TAB 3: EDIT (ROBUST) ---
with tab3:
    st.header("Manage Configs")
    ws = connect_to_gsheets()
    all_data = ws.get_all_values() if ws else []
    
    if all_data:
        edit_choice = st.selectbox("Select to Edit", [r[0] for r in all_data])
        row_idx = [r[0] for r in all_data].index(edit_choice) + 1
        current_json = all_data[row_idx - 1][1]
        
        try:
            data = json.loads(current_json)
            if isinstance(data, dict): data = [data] # Fix for single-object error
            
            df_edit = pd.DataFrame(data)
            # Flatten list options for editing
            if 'Options' in df_edit.columns:
                df_edit['Options'] = df_edit['Options'].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
            
            edited = st.data_editor(df_edit, num_rows="dynamic", use_container_width=True)
            
            if st.button("Save Changes"):
                new_list = []
                for _, row in edited.iterrows():
                    opts = [x.strip() for x in str(row.get('Options','')).split(',') if x.strip()]
                    new_list.append({"Column": row['Column'], "Type": row['Type'], "Options": opts})
                
                ws.update_cell(row_idx, 2, json.dumps(new_list))
                st.success("Updated!")
                st.rerun()
        except Exception as e:
            st.error(f"Error loading config: {e}")
