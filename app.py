import streamlit as st
import pandas as pd
import json
import base64
from io import BytesIO
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ==========================================
# 1. CONFIG & AUTHENTICATION
# ==========================================

st.set_page_config(page_title="Agency OS - Version 7", layout="wide")
st.title("Agency OS: AI Cataloging Automation")

# A. API Key Auth (Checks for both uppercase and lowercase styles)
api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai_key")
if not api_key:
    st.error("❌ OpenAI API Key missing. Please check .streamlit/secrets.toml")
    st.stop()

client = OpenAI(api_key=api_key)

# B. Google Sheets Auth
def connect_to_gsheets():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        # Handle different secrets structures (nested or flat)
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
        else:
            creds_dict = dict(st.secrets)
            
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client_gs = gspread.authorize(creds)
        # Open the database sheet
        return client_gs.open("Agency_OS_Database").worksheet("Configs")
    except Exception as e:
        st.error(f"⚠️ Database Connection Error: {e}")
        return None

# ==========================================
# 2. CORE AI LOGIC (Robust V7)
# ==========================================

def analyze_image_configured(image_bytes, config_json_str):
    """
    Takes an image and a JSON config.
    Returns a JSON dictionary extracted from the image based on rules.
    """
    # 1. Parse Configuration
    try:
        if isinstance(config_json_str, str):
            config = json.loads(config_json_str)
        else:
            config = config_json_str
            
        # SAFETY FIX: If config is a single dict (legacy error), wrap it in a list
        if isinstance(config, dict):
            config = [config]
            
    except json.JSONDecodeError:
        return {}

    # 2. Build Prompt based on Column Type
    ai_instructions = []
    
    for field in config:
        # Safety check for malformed rules
        if not isinstance(field, dict): continue
        
        if field.get("Type") == "AI":
            col = field.get("Column")
            options = field.get("Options", [])
            
            if options and isinstance(options, list) and len(options) > 0:
                # STRICT MODE: Force choice from dropdown
                opts_str = ", ".join([f"'{str(x).strip()}'" for x in options if pd.notna(x)])
                ai_instructions.append(f"- **{col}**: Choose strictly from [{opts_str}]")
            else:
                # CREATIVE MODE: Generate text (Description, Name, etc.)
                ai_instructions.append(f"- **{col}**: Generate a detailed, accurate value describing the visual product.")

    if not ai_instructions:
        return {}

    # 3. Construct System Prompt
    system_text = (
        "You are a Senior E-commerce QA Specialist. "
        "Your job is to inspect product images and extract attributes.\n"
        "RULES:\n"
        "1. Return ONLY valid JSON.\n"
        "2. If a list of options is provided, you MUST pick one exactly. Do not hallucinate.\n"
        "3. If visual evidence is missing, return 'N/A'."
    )
    
    user_text = (
        f"Analyze this product image and generate JSON for these fields:\n\n"
        + "\n".join(ai_instructions) + 
        "\n\nReturn the JSON object."
    )

    # 4. Call GPT-4o
    try:
        base64_img = base64.b64encode(image_bytes).decode('utf-8')
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]}
            ],
            max_tokens=600,
            temperature=0.2, # Low temp for strict adherence
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"Error": str(e)}

# ==========================================
# 3. USER INTERFACE (TABS)
# ==========================================

tab1, tab2, tab3 = st.tabs(["⚙️ Setup", "🚀 Run", "🗂️ Manage"])

# ------------------------------------------
# TAB 1: SETUP (Upload -> Map -> Save)
# ------------------------------------------
with tab1:
    st.header("1. Define Marketplace Rules")
    
    c1, c2 = st.columns(2)
    template_file = c1.file_uploader("Upload Blank Marketplace Template (Excel/CSV)", type=["xlsx", "csv"])
    master_file = c2.file_uploader("Upload Master Data / Dropdowns (Excel/CSV)", type=["xlsx", "csv"])

    # Store Master Options for linking
    master_options = {}
    
    # Process Master Data
    if master_file:
        try:
            df_master = pd.read_csv(master_file) if master_file.name.endswith('.csv') else pd.read_excel(master_file)
            # Basic cleaning
            for col in df_master.columns:
                master_options[col] = df_master[col].dropna().unique().tolist()
            st.success(f"✅ Loaded Dropdowns for: {', '.join(master_options.keys())}")
        except Exception as e:
            st.error(f"Error reading Master Data: {e}")

    # Process Template & Mapping
    if template_file:
        st.divider()
        st.subheader("2. Map Columns")
        try:
            df_temp = pd.read_csv(template_file) if template_file.name.endswith('.csv') else pd.read_excel(template_file)
            columns = df_temp.columns.tolist()
            
            config_list = []
            
            with st.form("config_form"):
                for col in columns:
                    c_a, c_b = st.columns([2, 1])
                    c_a.write(f"**{col}**")
                    field_type = c_b.selectbox("Type", ["Fixed", "Input", "AI"], key=f"type_{col}", label_visibility="collapsed")
                    
                    # Auto-link options if column name matches Master Data
                    opts = []
                    if field_type == "AI" and col in master_options:
                        opts = master_options[col]
                        c_a.caption(f"🔗 Linked to {len(opts)} options")
                    
                    config_list.append({
                        "Column": col,
                        "Type": field_type,
                        "Options": opts
                    })
                
                st.divider()
                cat_name = st.text_input("Configuration Name (e.g. 'Myntra Kurta')")
                submitted = st.form_submit_button("Save Configuration to Database")
                
                if submitted and cat_name:
                    ws = connect_to_gsheets()
                    if ws:
                        # Append row: [Name, JSON String]
                        ws.append_row([cat_name, json.dumps(config_list)])
                        st.success(f"🎉 Saved '{cat_name}' to Google Sheets!")
        except Exception as e:
            st.error(f"Error reading Template: {e}")

# ------------------------------------------
# TAB 2: RUN (Select -> Upload Images -> Excel)
# ------------------------------------------
with tab2:
    st.header("Generate Listings")
    
    ws = connect_to_gsheets()
    if ws:
        # Fetch Configs
        try:
            data_rows = ws.get_all_values()
        except:
            data_rows = []
            
        if not data_rows:
            st.warning("No configurations found in Database. Go to Setup tab.")
        else:
            # Column A is Name
            config_names = [row[0] for row in data_rows]
            selected_config = st.selectbox("Select Category", config_names)
            
            # Retrieve the JSON for the selected config
            config_json_str = next((row[1] for row in data_rows if row[0] == selected_config), None)
            
            uploaded_files = st.file_uploader("Upload Product Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
            
            if st.button("Start AI Processing"):
                if not uploaded_files:
                    st.warning("Please upload at least one image.")
                elif not config_json_str:
                    st.error("Configuration error.")
                else:
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, img_file in enumerate(uploaded_files):
                        # 1. Run AI
                        ai_data = analyze_image_configured(img_file.getvalue(), config_json_str)
                        
                        # 2. Map to Template Columns
                        row_data = {"Image Name": img_file.name}
                        
                        # Parse config to ensure we respect column order
                        try:
                            config_obj = json.loads(config_json_str)
                            if isinstance(config_obj, dict): config_obj = [config_obj] # Safety fix
                            
                            for rule in config_obj:
                                col = rule["Column"]
                                r_type = rule["Type"]
                                
                                if r_type == "AI":
                                    row_data[col] = ai_data.get(col, "")
                                elif r_type == "Fixed":
                                    row_data[col] = "FIXED_VALUE"
                                else:
                                    row_data[col] = "" # Input
                        except:
                            pass
                            
                        results.append(row_data)
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    # 3. Create DataFrame & Download
                    df_final = pd.DataFrame(results)
                    st.success("Processing Complete!")
                    st.dataframe(df_final)
                    
                    # Export to Excel
                    output = BytesIO()
                    # Using default engine to avoid missing dependency errors
                    with pd.ExcelWriter(output) as writer:
                        df_final.to_excel(writer, index=False)
                    output.seek(0)
                    
                    st.download_button(
                        label="Download Excel File",
                        data=output,
                        file_name=f"{selected_config}_Generated.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

# ------------------------------------------
# TAB 3: MANAGE (Cleanup)
# ------------------------------------------
with tab3:
    st.header("Database Management")
    
    if st.button("Refresh List"):
        st.rerun()

    ws = connect_to_gsheets()
    if ws:
        rows = ws.get_all_values()
        if rows:
            df_db = pd.DataFrame(rows, columns=["Category Name", "Config JSON"])
            st.dataframe(df_db)
            st.info("To delete, please delete the row directly in your Google Sheet.")
            st.markdown(f"[Open Google Sheet](https://docs.google.com/spreadsheets/d/{ws.spreadsheet.id})")
        else:
            st.write("Database is empty.")
