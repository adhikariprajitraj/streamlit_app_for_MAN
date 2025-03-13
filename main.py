import streamlit as st
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import os


def convert_to_dict(df, name_col=None, reg_col=None):
    """Create a dictionary mapping names to registration numbers.
    Handles different column name formats in different CSV files."""
    # Try to determine the name column if not specified
    if name_col is None:
        if 'Name' in df.columns:
            name_col = 'Name'
        elif 'Name of Students' in df.columns:
            name_col = 'Name of Students'
        else:
            st.error(f"Could not find name column in dataframe. Available columns: {df.columns.tolist()}")
            return {}
    
    # Try to determine the registration column if not specified
    if reg_col is None:
        if 'Registration No.' in df.columns:
            reg_col = 'Registration No.'
        elif 'Registration No' in df.columns:
            reg_col = 'Registration No'
        else:
            st.error(f"Could not find registration column in dataframe. Available columns: {df.columns.tolist()}")
            return {}
    
    # Create and return the dictionary
    try:
        return dict(zip(df[name_col], df[reg_col]))
    except KeyError as e:
        st.error(f"Error creating dictionary: Column '{e}' not found. Available columns: {df.columns.tolist()}")
        return {}


def load_data():
    """Load all required data files."""
    try:
        data = pd.read_csv('./MAN/final/DMO_final_score.csv')
        nmo = pd.read_csv("./NMO_result/NMO-Result.csv")
        top25 = pd.read_csv("./NMO_result/top25.csv")
        top100 = pd.read_csv("./NMO_result/top100.csv")
        dmo_2024 = pd.read_csv('./2024 result/DMO-2024.csv')
        pmo_2024 = pd.read_csv('./2024 result/PMO-2024.csv')
        nmo_2024 = pd.read_csv('./2024 result/NMO-2024.csv')
        tst_2024 = pd.read_csv('./2024 result/TST-2024.csv')
        
        # Debug information
        st.sidebar.expander("Debug Info", expanded=False).write({
            "DMO 2023 columns": data.columns.tolist(),
            "NMO columns": nmo.columns.tolist(),
            "DMO 2024 columns": dmo_2024.columns.tolist(),
            "PMO 2024 columns": pmo_2024.columns.tolist(),
            "NMO 2024 columns": nmo_2024.columns.tolist(),
            "TST 2024 columns": tst_2024.columns.tolist()
        })
        
        return data, nmo, top25, top100, dmo_2024, pmo_2024, nmo_2024, tst_2024
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None


def create_dicts(data, nmo, top25, top100, dmo_2024, pmo_2024, nmo_2024, tst_2024):
    """Create dictionaries for all datasets with appropriate column names."""
    # For each dataframe, determine the correct column names
    dmo_dict = convert_to_dict(data, 'Name of Students', 'Registration No.')
    dictionary_for_certificates = convert_to_dict(nmo, 'Name of Students', 'Registration No.')
    top25_dict = convert_to_dict(top25, 'Name of Students', 'Registration No.')
    top100_dict = convert_to_dict(top100, 'Name of Students', 'Registration No.')
    
    # 2024 data may have different column names
    dict_dmo_2024 = convert_to_dict(dmo_2024)
    dict_pmo_2024 = convert_to_dict(pmo_2024)
    dict_nmo_2024 = convert_to_dict(nmo_2024)
    dict_tst_2024 = convert_to_dict(tst_2024)
    
    return dmo_dict, dictionary_for_certificates, top25_dict, top100_dict, dict_dmo_2024, dict_pmo_2024, dict_nmo_2024, dict_tst_2024


def generate_certificate(name, font_path, certificate_path, student_dict, x_name=690, y_name=1150, x_symbol=750, y_symbol=710):
    """Generate certificate with given parameters."""
    # Ensure name is properly formatted
    name = name.title()
    
    if not student_dict:
        st.error("Student dictionary is empty. Check data loading.")
        return None
        
    if name in student_dict:
        try:
            # Load the certificate template
            image = Image.open(certificate_path)
            symbol_no = str(student_dict[name])
            draw = ImageDraw.Draw(image)
            
            # Set fonts
            font1 = ImageFont.truetype(font_path, 150)
            font2 = ImageFont.truetype(font_path, 70)
            
            # Handle long names
            if len(name) > 18:
                font1 = ImageFont.truetype(font_path, 130)
                x_name = x_name - 40  # Adjust position for long names
            
            # Add text to certificate
            draw.text((x_name, y_name), name, font=font1, fill=(0, 0, 0))
            draw.text((x_symbol, y_symbol), symbol_no, font=font2, fill=(0, 0, 0))
            
            # Save and return the image
            image_bytes = io.BytesIO()
            image.save(image_bytes, 'PNG')
            image_bytes.seek(0)
            return image_bytes
            
        except Exception as e:
            st.error(f"Error generating certificate: {e}")
            return None
    else:
        st.error(f"{name} is not in the list of students.")
        return None


def plot_distribution(score):
    sns.histplot(score, stat='density', color='#44eecc')
    plt.savefig(fname='plot')
    st.image('plot.png')
    os.remove('plot.png')


def show_stats(score):
    st.write(f'**Mean:**  {np.mean(score)}\n**Median:**  {np.median(score)}\n'
             f'**Standard Deviation:  **{np.std(score)}\n**Variance:**  {np.var(score)}\n'
             f'**Max: ** {np.max(score)} \n **Min:** {np.min(score)}')


def main():
    # Load data
    result = load_data()
    if result is None:
        st.error("Failed to load data. Please check file paths and try again.")
        return
        
    data, nmo, top25, top100, dmo_2024, pmo_2024, nmo_2024, tst_2024 = result
    
    # Extract score for statistics
    if 'Score' in data.columns:
        score = data['Score']
    else:
        st.warning("Score column not found in DMO data. Statistics functionality will be limited.")
        score = []
    
    # Create dictionaries for lookups
    dicts = create_dicts(data, nmo, top25, top100, dmo_2024, pmo_2024, nmo_2024, tst_2024)
    dmo_dict, dictionary_for_certificates, top25_dict, top100_dict, dict_dmo_2024, dict_pmo_2024, dict_nmo_2024, dict_tst_2024 = dicts

    # App UI
    st.title("Student Certificate Generator and Stats Viewer for IMO 2023 and 2024")
    st.caption("By Prajit Adhikari")

    menu = ["Generate Certificate for 2024 contests", "Home", "Generate Certificate for PreTST and TST for 2023 IMO", "View Statistics"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Generate Certificate for 2024 contests":
        st.subheader("Generate Certificate for 2024 IMO")
        
        # Create name column selector based on available columns
        name_col_dmo = 'Name' if 'Name' in dmo_2024.columns else 'Name of Students' if 'Name of Students' in dmo_2024.columns else dmo_2024.columns[0]
        name_col_nmo = 'Name' if 'Name' in nmo_2024.columns else 'Name of Students' if 'Name of Students' in nmo_2024.columns else nmo_2024.columns[0]
        name_col_tst = 'Name' if 'Name' in tst_2024.columns else 'Name of Students' if 'Name of Students' in tst_2024.columns else tst_2024.columns[0]
        
        student_name = st.selectbox("Select the name of the student (For DMO and PMO): ", sorted(dmo_2024[name_col_dmo]))
        last_year_student_name = st.selectbox("Select the name for NMO: ", sorted(nmo_2024[name_col_nmo]))
        tst_2024_student_name = st.selectbox("Select the name for TST: ", sorted(tst_2024[name_col_tst]))
        certificate_type = st.selectbox("Select certificate type", ["DMO", "PMO", "NMO", "TST"])
        
        if st.button("Generate"):
            image_bytes = None
            download_name = ""
            
            try:
                if certificate_type == "DMO":
                    image_bytes = generate_certificate(student_name, "COMIC.TTF", "./2024_certificates/certificate for DMO.png", dict_dmo_2024, 600, 1100, 750, 710)
                    download_name = student_name
                elif certificate_type == "PMO":
                    image_bytes = generate_certificate(student_name, "COMIC.TTF", "./2024_certificates/certificate for PMO.png", dict_pmo_2024, 650, 1150, 750, 710)
                    download_name = student_name
                elif certificate_type == "NMO":
                    image_bytes = generate_certificate(last_year_student_name, "COMIC.TTF", "./2024_certificates/certificate for NMO.png", dict_nmo_2024, 650, 1150, 750, 710)
                    download_name = last_year_student_name
                elif certificate_type == "TST":
                    image_bytes = generate_certificate(tst_2024_student_name, "COMIC.TTF", "./2024_certificates/certificate for TST.png", dict_tst_2024, 650, 1150, 750, 710)
                    download_name = tst_2024_student_name
                
                if image_bytes is not None:
                    st.image(image_bytes, caption='Generated certificate')
                    st.download_button(
                        "Download Certificate",
                        data=image_bytes,
                        file_name=f'{download_name}_{certificate_type}_2024_certificate.png',
                        mime='image/png'
                    )
            except Exception as e:
                st.error(f"Error in certificate generation: {e}")

    elif choice == "Home":
        st.subheader("Home")
        st.write("Welcome to the Student Certificate Generator and Stats Viewer. Please select an action from the sidebar.")
        st.subheader("Generate Certificate")
        
        # Use proper column name for student selection
        name_col = 'Name of Students' if 'Name of Students' in data.columns else 'Name'
        student_name = st.selectbox("Select the name of the student: ", sorted(data[name_col].unique()))
        
        certificate_type = st.selectbox("Select certificate type", ["DMO", "NMO"])
        if st.button("Generate"):
            if certificate_type == "DMO":
                image_bytes = generate_certificate(student_name, "COMIC.TTF", "./for_certificates/certificate for DMO.png", dmo_dict, 600, 1100, 750, 710)
            elif certificate_type == "NMO":
                image_bytes = generate_certificate(student_name, "COMIC.TTF", "./for_certificates/certificate for NMO.png", dictionary_for_certificates, 650, 1150, 750, 710)
            
            if image_bytes is not None:
                st.image(image_bytes, caption='Generated certificate')
                st.download_button(
                    "Download Certificate",
                    data=image_bytes,
                    file_name=f'{student_name}_{certificate_type}_2023_certificate.png',
                    mime='image/png'
                )

    elif choice == "Generate Certificate for PreTST and TST for 2023 IMO":
        st.subheader("Generate Certificate for PreTST and TST 2023 IMO")
        
        # Use proper column name for student selection
        name_col = 'Name of Students' if 'Name of Students' in top100.columns else 'Name'
        student_name = st.selectbox("Select the name of the student: ", sorted(top100[name_col]))
        
        certificate_type = st.selectbox("Select certificate type", ["Pre-TST", "TST"])
        if st.button("Generate"):
            if certificate_type == "Pre-TST":
                image_bytes = generate_certificate(student_name, "COMIC.TTF", "./for_certificates/pretst2023.png", top100_dict, 750, 1150, 750, 710)
            elif certificate_type == "TST":
                image_bytes = generate_certificate(student_name, "COMIC.TTF", "./for_certificates/TST round certificate.png", top25_dict, 750, 1150, 750, 710)

            if image_bytes is not None:
                st.image(image_bytes, caption='Generated certificate')
                st.download_button(
                    "Download Certificate",
                    data=image_bytes,
                    file_name=f'{student_name}_{certificate_type}_2023_certificate.png',
                    mime='image/png'
                )

    elif choice == "View Statistics":
        st.subheader("View Statistics")
        if len(score) > 0:
            if st.button("Show Distribution"):
                plot_distribution(score)
            if st.button("Show Stats"):
                show_stats(score)
        else:
            st.warning("Statistics not available due to missing score data.")


if __name__ == "__main__":
    main()
