import streamlit as st
import pandas as pd
import json
import base64
from io import BytesIO
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================

st.set_page_config(page_title="Agency OS - Project M", layout="wide")
st.title("Agency OS: AI Cataloging Automation")

# API Key Check (Checks for both naming conventions)
api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai_key")
if api_key:
    client = OpenAI(api_key=api_key)
else:
    st.error("❌ API Key Error: Please make sure your .streamlit/secrets.toml has OPENAI_API_KEY")
    st.stop()

# Google Sheets Connection
def connect_to_gsheets():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
        else:
            creds_dict = dict(st.secrets)
            
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client_gs = gspread.authorize(creds)
        sheet = client_gs.open("Agency_OS_Database")
        return sheet.worksheet("Configs")
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return None

# ==========================================
# 2. CORE AI LOGIC (Dual-Mode)
# ==========================================

def analyze_image_configured(image_bytes, config_json, seo_keywords=""):
    """
    Analyzes an image using GPT-4o with Dual-Mode logic (Strict vs Creative).
    """
    if isinstance(config_json, str):
        try:
            config_json = json.loads(config_json)
        except:
            return {}

    strict_fields = []
    creative_fields = []

    for field in config_json:
        if field.get("Type") == "AI":
            col_name = field.get("Column")
            options = field.get("Options", [])
            
            # STRICT MODE: If options exist, force selection
            if isinstance(options, list) and len(options) > 0:
                clean_opts = [str(opt).strip() for opt in options if pd.notna(opt)]
                options_str = ", ".join([f"'{opt}'" for opt in clean_opts])
                strict_fields.append(f"- **{col_name}**: Choose strictly from [{options_str}]")
            # CREATIVE MODE: If no options, write text
            else:
                col_lower = col_name.lower()
                if "tag" in col_lower or "keyword" in col_lower:
                    guide = f"Generate 10-15 high-traffic SEO comma-separated tags. Context: {seo_keywords}."
                elif "name" in col_lower or "title" in col_lower:
                    guide = f"Generate a Product Display Name (Brand + Style + Material). Context: {seo_keywords}."
                elif "tip" in col_lower or "wear" in col_lower:
                    guide = f"Style tip & Occasion. Context: {seo_keywords}."
                else:
                    guide = f"Detailed description. Context: {seo_keywords}."
                creative_fields.append(f"- **{col_name}**: {guide}")

    if not strict_fields and not creative_fields:
        return {}

    prompt_sections = []
    if strict_fields:
        prompt_sections.append("### PART 1: STRICT CLASSIFICATION (Use exact dropdown value)\n" + "\n".join(strict_fields))
    if creative_fields:
        prompt_sections.append("### PART 2: CREATIVE SEO (Write engaging text)\n" + "\n".join(creative_fields))

    system_prompt = "You are an AI Cataloging Expert. Output ONLY valid JSON."
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
# 3. USER INTERFACE
# ==========================================

tab1, tab2, tab3 = st.tabs(["⚙️ Setup (Create)", "🚀 Run (Batch Process)", "✏️ Edit Configs"])

# ------------------------------------------
# TAB 1: SETUP (Create New)
# ------------------------------------------
with tab1:
    st.header("Create New Configuration")
    c1, c2 = st.columns(2)
    with c1:
        template_file = st.file_uploader("1. Upload Blank Template", type=["xlsx", "csv"])
    with c2:
        master_file = st.file_uploader("2. Upload Master Data (Dropdowns)", type=["xlsx", "csv"])

    final_master_options = {}

    if master_file:
        st.subheader("Edit Dropdown Options")
        df_master = pd.read_csv(master_file) if master_file.name.endswith('.csv') else pd.read_excel(master_file)
        edited_df = st.data_editor(df_master, num_rows="dynamic") # Edit feature
        for col in edited_df.columns:
            final_master_options[col] = edited_df[col].dropna().unique().tolist()

    if template_file:
        st.subheader("Map Columns")
        df_temp = pd.read_csv(template_file) if template_file.name.endswith('.csv') else pd.read_excel(template_file)
        columns = df_temp.columns.tolist()
        
        config_builder = []
        with st.form("mapping_form"):
            for col in columns:
                c_a, c_b = st.columns([2, 1])
                with c_a: st.write(f"**{col}**")
                with c_b: type_ = st.selectbox("Type", ["Fixed", "Input", "AI"], key=f"t_{col}", label_visibility="collapsed")
                
                field_data = {"Column": col, "Type": type_, "Options": []}
                if type_ == "AI" and col in final_master_options:
                    field_data["Options"] = final_master_options[col]
                config_builder.append(field_data)
            
            st.divider()
            name = st.text_input("Configuration Name")
            if st.form_submit_button("Save"):
                ws = connect_to_gsheets()
                ws.append_row([name, json.dumps(config_builder)])
                st.success("Saved!")

# ------------------------------------------
# TAB 2: RUN (Input Excel + Images)
# ------------------------------------------
with tab2:
    st.header("Batch Generator")
    ws = connect_to_gsheets()
    
    # 1. Select Config
    all_data = ws.get_all_values()
    config_names = [r[0] for r in all_data] if all_data else []
    selected_conf = st.selectbox("Select Category", config_names)
    
    # 2. Upload Files
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        input_excel = st.file_uploader("1. Upload Input Excel (Must have 'Image Name' column)", type=['xlsx'])
    with col_up2:
        input_images = st.file_uploader("2. Upload Images", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])

    # 3. SEO Context
    seo_keywords = st.text_area("SEO Keywords", placeholder="e.g. Summer, Cotton, Party Wear")

    if st.button("Generate Catalog") and input_excel and input_images:
        # Get Config JSON
        json_str = next((r[1] for r in all_data if r[0] == selected_conf), "[]")
        
        # Load Input Excel
        df_input = pd.read_excel(input_excel)
        
        # Check Image Column
        # We look for "Image Name" or "Image_Name" or just "Image"
        img_col = next((c for c in df_input.columns if "image" in c.lower() and "name" in c.lower()), None)
        
        if not img_col:
            st.error("❌ Input Excel must have a column named 'Image Name'.")
        else:
            # Map Uploaded Images: Filename -> Bytes
            image_map = {img.name: img.getvalue() for img in input_images}
            
            st.info(f"Found {len(df_input)} rows in Excel and {len(image_map)} images uploaded.")
            
            progress = st.progress(0)
            
            # Iterate Rows
            for index, row in df_input.iterrows():
                img_name = str(row[img_col]).strip()
                
                if img_name in image_map:
                    # Run AI
                    ai_result = analyze_image_configured(image_map[img_name], json_str, seo_keywords)
                    
                    # Fill Data Frame
                    for key, value in ai_result.items():
                        if key in df_input.columns:
                            df_input.at[index, key] = value
                
                progress.progress((index + 1) / len(df_input))
            
            # Export
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_input.to_excel(writer, index=False)
            output.seek(0)
            
            st.success("Processing Complete!")
            st.download_button("Download Completed Excel", data=output, file_name="Completed_Catalog.xlsx")

# ------------------------------------------
# TAB 3: EDIT CONFIGS (Fixed & Robust)
# ------------------------------------------
with tab3:
    st.header("Edit Existing Configurations")
    
    ws = connect_to_gsheets()
    if ws:
        all_data = ws.get_all_values()
    else:
        all_data = []
    
    if len(all_data) > 0:
        # Select Config to Edit
        config_names = [r[0] for r in all_data]
        edit_choice = st.selectbox("Select Configuration to Edit", config_names)
        
        # Find Row Index (1-based for GSheets)
        row_idx = config_names.index(edit_choice) + 1
        current_json_str = all_data[row_idx - 1][1]
        
        if current_json_str:
            try:
                current_config = json.loads(current_json_str)
                
                # === BUG FIX START ===
                # If the data is a single Dictionary (One rule), wrap it in a list.
                # If the data is a List (Multiple rules), keep it as is.
                if isinstance(current_config, dict):
                    current_config = [current_config]
                # === BUG FIX END ===
                
                df_config = pd.DataFrame(current_config)
                
                st.subheader(f"Editing: {edit_choice}")
                st.info("👇 You can change Types or Edit Options. For 'Options', enter values separated by commas.")
                
                # Helper to display options as string for editing
                # We use .get('Options') to avoid errors if the column is missing
                if 'Options' in df_config.columns:
                    df_config['Options'] = df_config['Options'].apply(
                        lambda x: ", ".join(x) if isinstance(x, list) else str(x) if pd.notna(x) else ""
                    )
                else:
                    df_config['Options'] = ""
                
                # Editable Dataframe
                edited_df = st.data_editor(df_config, num_rows="dynamic", use_container_width=True)
                
                if st.button("Save Changes"):
                    # Convert back to JSON structure
                    new_config_list = []
                    for _, row in edited_df.iterrows():
                        # Parse options back to list
                        opts_str = str(row.get('Options', ''))
                        opts_list = [x.strip() for x in opts_str.split(',') if x.strip()]
                        
                        new_config_list.append({
                            "Column": row.get('Column', 'Unknown'),
                            "Type": row.get('Type', 'Fixed'),
                            "Options": opts_list
                        })
                    
                    new_json_str = json.dumps(new_config_list)
                    
                    # Update Google Sheet (Row, Col 2)
                    ws.update_cell(row_idx, 2, new_json_str)
                    st.success(f"✅ Configuration '{edit_choice}' updated successfully!")
                    
                    # Optional: Rerun to show updated state
                    # st.rerun() 
                    
            except json.JSONDecodeError:
                st.error("❌ Error: The saved data for this category is not valid JSON.")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
                st.write("Raw Data for debugging:", current_config)
