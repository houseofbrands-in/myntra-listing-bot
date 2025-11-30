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

# ERROR FIX: Check for the key using the name from your screenshot
# We check both "OPENAI_API_KEY" (what you have) and "openai_key" (what I wrote before)
api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai_key")

if api_key:
    client = OpenAI(api_key=api_key)
else:
    st.error("❌ API Key Error: Please make sure your .streamlit/secrets.toml contains:")
    st.code('OPENAI_API_KEY = "sk-..."', language="toml")
    st.stop()

# Initialize Google Sheets Connection
def connect_to_gsheets():
    try:
        # Create a dict from the secrets object for credentials
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        
        # Check if user put it in [gcp_service_account] section or root
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
        else:
            # Fallback: maybe they pasted the JSON keys directly at the root
            creds_dict = dict(st.secrets)
            
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # Open the specific Sheet
        sheet = client.open("Agency_OS_Database")
        worksheet = sheet.worksheet("Configs")
        return worksheet
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return None
# ==========================================
# 2. CORE AI LOGIC (Dual-Mode)
# ==========================================

def analyze_image_configured(image_bytes, config_json, seo_keywords=""):
    """
    Analyzes an image using GPT-4o with Dual-Mode logic:
    1. STRICT Mode: For columns that have 'Options' (Dropdowns).
    2. CREATIVE Mode: For columns without options (Description, SEO Tags, Title).
    """
    
    # Segregate Fields
    strict_fields = []
    creative_fields = []
    
    # We expect config_json to be a Python list (parsed from JSON string already if needed)
    if isinstance(config_json, str):
        try:
            config_json = json.loads(config_json)
        except:
            return {}

    for field in config_json:
        if field.get("Type") == "AI":
            col_name = field.get("Column")
            options = field.get("Options", [])
            
            # Check if options exist and represent a list
            if isinstance(options, list) and len(options) > 0:
                # STRICT MODE (Dropdowns)
                # Clean options to ensure strings
                clean_opts = [str(opt).strip() for opt in options if pd.notna(opt)]
                options_str = ", ".join([f"'{opt}'" for opt in clean_opts])
                strict_fields.append(f"- **{col_name}**: Choose strictly from [{options_str}]")
            else:
                # CREATIVE MODE (SEO/Text)
                col_lower = col_name.lower()
                if "tag" in col_lower or "keyword" in col_lower:
                    guide = f"Generate 10-15 high-traffic SEO comma-separated tags. Use input context: {seo_keywords}."
                elif "name" in col_lower or "title" in col_lower:
                    guide = f"Generate a click-worthy Product Display Name (Brand + Style + Material). Context: {seo_keywords}."
                elif "tip" in col_lower or "wear" in col_lower:
                    guide = f"Write a fashion style tip. Mention occasions. Context: {seo_keywords}."
                else:
                    guide = f"Write a detailed description optimizing for search. Context: {seo_keywords}."
                
                creative_fields.append(f"- **{col_name}**: {guide}")

    if not strict_fields and not creative_fields:
        return {}

    # Construct Prompt
    prompt_sections = []
    if strict_fields:
        prompt_sections.append("### PART 1: STRICT CLASSIFICATION (Data Integrity)\n" + "\n".join(strict_fields))
    if creative_fields:
        prompt_sections.append("### PART 2: CREATIVE SEO GENERATION (Discoverability)\n" + "\n".join(creative_fields))

    system_prompt = (
        "You are an AI Expert for Myntra/Flipkart Cataloging. "
        "1. For strict fields, pick the EXACT value from the list. Do not hallucinate.\n"
        "2. For creative fields, write engaging, high-SEO content.\n"
        "3. Output ONLY valid JSON."
    )

    user_prompt = (
        f"Analyze this image and generate JSON.\n\n" + 
        "\n\n".join(prompt_sections) + 
        "\n\nReturn JSON object."
    )

    # Call OpenAI
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
# 3. USER INTERFACE (TABS)
# ==========================================

tab1, tab2, tab3 = st.tabs(["⚙️ Setup (Create Rules)", "🚀 Run (Generate)", "🗂️ Manage Configs"])

# ------------------------------------------
# TAB 1: SETUP (Save to Google Sheets)
# ------------------------------------------
with tab1:
    st.header("Create New Category Configuration")
    
    c1, c2 = st.columns(2)
    with c1:
        # Upload Blank Template (Structure)
        template_file = st.file_uploader("1. Upload Blank Marketplace Template (Excel)", type=["xlsx", "csv"])
    with c2:
        # Upload Master Data (Dropdowns)
        master_file = st.file_uploader("2. Upload Master Data / Dropdowns (Excel)", type=["xlsx", "csv"])

    if template_file:
        # Load Template Headers
        if template_file.name.endswith('.csv'):
            df_temp = pd.read_csv(template_file)
        else:
            df_temp = pd.read_excel(template_file)
        
        columns = df_temp.columns.tolist()
        
        # Load Master Data Dictionary
        master_options = {}
        if master_file:
            if master_file.name.endswith('.csv'):
                df_master = pd.read_csv(master_file)
            else:
                df_master = pd.read_excel(master_file)
            
            # Convert Master Data to dict: {ColumnName: [Option1, Option2...]}
            for col in df_master.columns:
                # Get unique values, drop NaNs
                opts = df_master[col].dropna().unique().tolist()
                master_options[col] = opts
            
            st.success(f"Loaded Master Data for columns: {', '.join(master_options.keys())}")

        st.divider()
        st.subheader("Map Your Columns")
        
        # Mapping Logic
        config_builder = []
        
        # Use a form to prevent reload on every click
        with st.form("mapping_form"):
            for col in columns:
                c_a, c_b = st.columns([2, 1])
                with c_a:
                    st.write(f"**{col}**")
                with c_b:
                    # Determine type
                    field_type = st.selectbox(f"Type for {col}", ["Fixed", "Input", "AI"], key=f"type_{col}")
                
                field_data = {"Column": col, "Type": field_type, "Options": []}
                
                # If AI and Master Data exists for this column, attach options automatically
                if field_type == "AI" and col in master_options:
                    field_data["Options"] = master_options[col]
                    st.caption(f"✅ Linked {len(field_data['Options'])} dropdown options")
                
                config_builder.append(field_data)
            
            # Save Config Section
            st.divider()
            category_name = st.text_input("Name this Category Configuration (e.g., 'Myntra Kurta')")
            submitted = st.form_submit_button("Save Configuration to Database")
            
            if submitted and category_name:
                json_config = json.dumps(config_builder)
                ws = connect_to_gsheets()
                if ws:
                    # Append row: [Category Name, JSON Config]
                    ws.append_row([category_name, json_config])
                    st.success(f"Configuration '{category_name}' saved to Google Sheets!")

# ------------------------------------------
# TAB 2: RUN (Fetch from GSheets & Process)
# ------------------------------------------
with tab2:
    st.header("Generate Listings")
    
    ws = connect_to_gsheets()
    if ws:
        # Fetch all existing configs
        data = ws.get_all_values() # Returns list of lists
        # Skip header if row 1 is header, assuming data starts
        # Format: [[Name, JSON], [Name, JSON]]
        
        if len(data) > 0:
            config_names = [row[0] for row in data]
            selected_config_name = st.selectbox("Select Category", config_names)
            
            # Find the JSON for selected category
            selected_json_str = next((row[1] for row in data if row[0] == selected_config_name), None)
            
            if selected_json_str:
                # SEO Input
                seo_keywords = st.text_area("SEO Keywords / Context (Optional)", 
                                           placeholder="e.g. Summer Collection, Cotton, Breathable, Ethnic Wear, Diwali Sale")
                
                # Image Upload
                uploaded_images = st.file_uploader("Upload Product Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
                
                if st.button("Start AI Processing"):
                    if not uploaded_images:
                        st.warning("Please upload images.")
                    else:
                        results_list = []
                        progress_bar = st.progress(0)
                        
                        for idx, img_file in enumerate(uploaded_images):
                            bytes_data = img_file.getvalue()
                            
                            # Call the DUAL MODE AI function
                            ai_data = analyze_image_configured(bytes_data, selected_json_str, seo_keywords)
                            
                            # Build the row
                            row_data = {"Image Name": img_file.name}
                            
                            # Merge AI data into row based on config
                            config_obj = json.loads(selected_json_str)
                            for field in config_obj:
                                col = field["Column"]
                                f_type = field["Type"]
                                
                                if f_type == "AI":
                                    # Get from AI response, default to empty
                                    row_data[col] = ai_data.get(col, "")
                                elif f_type == "Fixed":
                                    row_data[col] = "FIXED_VALUE" # Placeholder
                                else:
                                    row_data[col] = "" # Input type left blank
                            
                            results_list.append(row_data)
                            progress_bar.progress((idx + 1) / len(uploaded_images))
                        
                        # Create DataFrame
                        df_result = pd.DataFrame(results_list)
                        st.success("Processing Complete!")
                        st.dataframe(df_result)
                        
                        # Download Button
                        # Convert to Excel in memory
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_result.to_excel(writer, index=False)
                        output.seek(0)
                        
                        st.download_button(
                            label="Download Excel File",
                            data=output,
                            file_name=f"{selected_config_name}_Generated.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

# ------------------------------------------
# TAB 3: MANAGE (Cleanup)
# ------------------------------------------
with tab3:
    st.header("Database Management")
    if st.button("Refresh Database View"):
        st.rerun()
        
    ws = connect_to_gsheets()
    if ws:
        data = ws.get_all_values()
        if len(data) > 0:
            df_db = pd.DataFrame(data, columns=["Category Name", "Configuration JSON"])
            st.dataframe(df_db)
            
            st.warning("To delete configurations, please manually delete rows in your Google Sheet 'Agency_OS_Database'.")
            st.markdown(f"[Open Google Sheet](https://docs.google.com/spreadsheets/d/{ws.spreadsheet.id})")

