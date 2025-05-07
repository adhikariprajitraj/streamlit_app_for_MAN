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
        # Load previous years' data
        data = pd.read_csv('./MAN/final/DMO_final_score.csv')
        nmo = pd.read_csv("./NMO_result/NMO-Result.csv")
        top25 = pd.read_csv("./NMO_result/top25.csv")
        top100 = pd.read_csv("./NMO_result/top100.csv")
        dmo_2024 = pd.read_csv('./2024 result/DMO-2024.csv')
        pmo_2024 = pd.read_csv('./2024 result/PMO-2024.csv')
        nmo_2024 = pd.read_csv('./2024 result/NMO-2024.csv')
        tst_2024 = pd.read_csv('./2024 result/TST-2024.csv')
        
        # Load 2025 data - Province-wise DMO and special categories
        try:
            # Load province-wise DMO data
            man_2025_all = {}
            province_files = {
                'Bagmati': './2025 result/csv_files/Bagmati.csv',
                'Gandaki': './2025 result/csv_files/Gandaki.csv',
                'Karnali': './2025 result/csv_files/Karnali.csv',
                'Koshi': './2025 result/csv_files/Koshi.csv',
                'KTM': './2025 result/csv_files/KTM.csv',
                'Lumbini': './2025 result/csv_files/Lumbini.csv',
                'Madhesh': './2025 result/csv_files/Madhesh.csv',
                'Sudurpaschim': './2025 result/csv_files/Sudurpaschim.csv'
            }
            
            # Load each province's data
            for province, file_path in province_files.items():
                try:
                    df = pd.read_csv(file_path)
                    # Clean up column names and data
                    if 'Name of the student' in df.columns:
                        df = df.rename(columns={'Name of the student': 'Name'})
                    elif 'Name of Students' in df.columns:
                        df = df.rename(columns={'Name of Students': 'Name'})
                    
                    # Convert name column to title case
                    name_col = 'Name' if 'Name' in df.columns else df.columns[2]
                    df[name_col] = df[name_col].str.strip().str.title()
                    
                    man_2025_all[province] = df
                except Exception as e:
                    st.sidebar.warning(f"Failed to load {province} data: {e}")
            
            # Load special categories
            pmo_2025 = pd.read_csv('./2025 result/csv_files/PMO.csv')
            ptst_2025 = pd.read_csv('./2025 result/csv_files/PTST.csv')
            tst_2025 = pd.read_csv('./2025 result/csv_files/TST.csv')
            
            # Clean and format special category data
            for df in [pmo_2025, ptst_2025, tst_2025]:
                name_col = 'Name' if 'Name' in df.columns else df.columns[2]
                df[name_col] = df[name_col].str.strip().str.title()
            
            # Add special categories to the dictionary
            man_2025_all['PMO'] = pmo_2025
            man_2025_all['PTST'] = ptst_2025
            man_2025_all['TST'] = tst_2025
            
            if man_2025_all:
                st.sidebar.success(f"Loaded 2025 data: {list(man_2025_all.keys())}")
            else:
                st.sidebar.warning("No 2025 data found")
                
        except Exception as e:
            st.sidebar.warning(f"2025 data load failed: {e}")
            man_2025_all = {}
        
        # Debug information
        st.sidebar.expander("Debug Info", expanded=False).write({
            "DMO 2023 columns": data.columns.tolist(),
            "NMO columns": nmo.columns.tolist(),
            "DMO 2024 columns": dmo_2024.columns.tolist(),
            "PMO 2024 columns": pmo_2024.columns.tolist(),
            "NMO 2024 columns": nmo_2024.columns.tolist(),
            "TST 2024 columns": tst_2024.columns.tolist(),
            "2025 available data": list(man_2025_all.keys()) if man_2025_all else "none"
        })
        
        return data, nmo, top25, top100, dmo_2024, pmo_2024, nmo_2024, tst_2024, man_2025_all
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None


def create_dicts(data, nmo, top25, top100, dmo_2024, pmo_2024, nmo_2024, tst_2024, man_2025_all):
    """Create dictionaries for all datasets with appropriate column names."""
    # Previous years' data
    dmo_dict = convert_to_dict(data, 'Name of Students', 'Registration No.')
    dictionary_for_certificates = convert_to_dict(nmo, 'Name of Students', 'Registration No.')
    top25_dict = convert_to_dict(top25, 'Name of Students', 'Registration No.')
    top100_dict = convert_to_dict(top100, 'Name of Students', 'Registration No.')
    
    # 2024 data
    dict_dmo_2024 = convert_to_dict(dmo_2024)
    dict_pmo_2024 = convert_to_dict(pmo_2024)
    dict_nmo_2024 = convert_to_dict(nmo_2024)
    dict_tst_2024 = convert_to_dict(tst_2024)
    
    # 2025 data - handle each province and special category
    man_2025_dicts = {}
    for category, df in man_2025_all.items():
        # Handle different column names
        name_col = ('Name' if 'Name' in df.columns 
                   else 'Name of Students' if 'Name of Students' in df.columns
                   else 'Name of the student' if 'Name of the student' in df.columns
                   else df.columns[2])  # Assuming name is in the third column if not found
        reg_col = ('Registration No.' if 'Registration No.' in df.columns
                  else 'Registration No' if 'Registration No' in df.columns
                  else df.columns[1])  # Assuming registration is in the second column if not found
        
        man_2025_dicts[category] = convert_to_dict(df, name_col, reg_col)
    
    return (dmo_dict, dictionary_for_certificates, top25_dict, top100_dict,
            dict_dmo_2024, dict_pmo_2024, dict_nmo_2024, dict_tst_2024,
            man_2025_dicts)


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
        
    data, nmo, top25, top100, dmo_2024, pmo_2024, nmo_2024, tst_2024, man_2025_all = result
    
    # Extract score for statistics
    if 'Score' in data.columns:
        score = data['Score']
    else:
        st.warning("Score column not found in DMO data. Statistics functionality will be limited.")
        score = []
    
    # Create dictionaries for lookups
    dicts = create_dicts(data, nmo, top25, top100, dmo_2024, pmo_2024, nmo_2024, tst_2024, man_2025_all)
    (dmo_dict, dictionary_for_certificates, top25_dict, top100_dict,
     dict_dmo_2024, dict_pmo_2024, dict_nmo_2024, dict_tst_2024,
     man_2025_dicts) = dicts

    # App UI
    st.title("Student Certificate Generator and Stats Viewer for IMO 2023-2025")
    st.caption("By Prajit Adhikari")

    menu = ["Generate Certificate for 2025 contests",
            "Generate Certificate for 2024 contests",
            "Home",
            "Generate Certificate for PreTST and TST for 2023 IMO",
            "View Statistics"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Generate Certificate for 2025 contests":
        st.subheader("Generate Certificate for 2025 IMO")
        
        # Separate provinces from special categories
        provinces = sorted([p for p in man_2025_dicts.keys() if p not in ['PMO', 'PTST', 'TST']])
        special_categories = ['PMO', 'PTST', 'TST']
        
        cert_type = st.radio("Certificate Type", ["DMO by Province", "PMO/PTST/TST"])
        
        if cert_type == "DMO by Province":
            province = st.selectbox("Select Province:", provinces)
            if province in man_2025_all:
                df_prov = man_2025_all[province]
                name_col = ('Name' if 'Name' in df_prov.columns
                           else 'Name of Students' if 'Name of Students' in df_prov.columns
                           else 'Name of the student' if 'Name of the student' in df_prov.columns
                           else df_prov.columns[2])
                
                # Get names, ensure they're in title case, strip whitespace, and sort
                names = sorted(df_prov[name_col].str.strip().str.title().unique())
                student_name = st.selectbox("Select Student:", names)
                
                # Map province names to certificate filenames
                cert_map = {
                    'Koshi': 'DMO 2025 KOshi.png',
                    'Madhesh': 'DMO 2025 Madesh.png',
                    'Bagmati': 'DMO 2025 Bagmati.png',
                    'Gandaki': 'DMO 2025 Gandki.png',
                    'Lumbini': 'DMO 2025 Lumbini.png',
                    'Karnali': 'DMO 2025 Karnali.png',
                    'Sudurpaschim': 'DMO 2025 Sudurpaschim.png',
                    'KTM': 'DMO 2025 Kathmandu valley.png'
                }
                
                if st.button("Generate"):
                    cert_path = f"./2025_certificates/{cert_map.get(province, 'certificate for PMO.png')}"
                    if not os.path.exists(cert_path):
                        st.error(f"Certificate template not found for {province}")
                    else:
                        img = generate_certificate(
                            student_name,
                            "COMIC.TTF",
                            cert_path,
                            man_2025_dicts[province],
                            600, 1100, 650, 710
                        )
                        if img:
                            st.image(img, caption=f"{province} DMO Certificate")
                            st.download_button(
                                "Download Certificate",
                                data=img,
                                file_name=f"{student_name}_{province}_DMO_2025.png",
                                mime="image/png"
                            )
        
        else:  # PMO/PTST/TST
            category = st.selectbox("Select Category:", special_categories)
            if category in man_2025_all:
                df_cat = man_2025_all[category]
                name_col = ('Name' if 'Name' in df_cat.columns
                           else 'Name of Students' if 'Name of Students' in df_cat.columns
                           else df_cat.columns[2])
                
                # Get names, ensure they're in title case, strip whitespace, and sort
                names = sorted(df_cat[name_col].str.strip().str.title().unique())
                student_name = st.selectbox("Select Student:", names)
                
                cert_templates = {
                    'PMO': 'certificate for PMO.png',
                    'PTST': 'certificate for pretst.png',
                    'TST': 'Certificate for TST.png'
                }
                
                if st.button("Generate"):
                    cert_path = f"./2025_certificates/{cert_templates[category]}"
                    if not os.path.exists(cert_path):
                        st.error(f"Certificate template not found for {category}")
                    else:
                        img = generate_certificate(
                            student_name,
                            "COMIC.TTF",
                            cert_path,
                            man_2025_dicts[category],
                            600, 1100, 650, 710
                        )
                        if img:
                            st.image(img, caption=f"{category} Certificate")
                            st.download_button(
                                "Download Certificate",
                                data=img,
                                file_name=f"{student_name}_{category}_2025.png",
                                mime="image/png"
                            )

    elif choice == "Generate Certificate for 2024 contests":
        st.subheader("Generate Certificate for 2024 IMO")
        
        # Create name column selector based on available columns
        name_col_dmo = 'Name' if 'Name' in dmo_2024.columns else 'Name of Students' if 'Name of Students' in dmo_2024.columns else dmo_2024.columns[0]
        name_col_nmo = 'Name' if 'Name' in nmo_2024.columns else 'Name of Students' if 'Name of Students' in nmo_2024.columns else nmo_2024.columns[0]
        name_col_tst = 'Name' if 'Name' in tst_2024.columns else 'Name of Students' if 'Name of Students' in tst_2024.columns else tst_2024.columns[0]
        
        # Get names, ensure they're in title case, strip whitespace, and sort
        dmo_names = sorted(dmo_2024[name_col_dmo].str.strip().str.title().unique())
        nmo_names = sorted(nmo_2024[name_col_nmo].str.strip().str.title().unique())
        tst_names = sorted(tst_2024[name_col_tst].str.strip().str.title().unique())
        
        student_name = st.selectbox("Select the name of the student (For DMO and PMO): ", dmo_names)
        last_year_student_name = st.selectbox("Select the name for NMO: ", nmo_names)
        tst_2024_student_name = st.selectbox("Select the name for TST: ", tst_names)
        certificate_type = st.selectbox("Select certificate type", ["DMO", "PMO", "NMO", "TST"])
        
        if st.button("Generate"):
            image_bytes = None
            download_name = ""
            
            try:
                if certificate_type == "DMO":
                    # Reduced x_symbol from 750 to 650
                    image_bytes = generate_certificate(student_name, "COMIC.TTF", "./2024_certificates/certificate for DMO.png", dict_dmo_2024, 600, 1100, 650, 710)
                    download_name = student_name
                elif certificate_type == "PMO":
                    # Reduced x_symbol from 750 to 650
                    image_bytes = generate_certificate(student_name, "COMIC.TTF", "./2024_certificates/certificate for PMO.png", dict_pmo_2024, 650, 1150, 650, 710)
                    download_name = student_name
                elif certificate_type == "NMO":
                    # Reduced x_symbol from 750 to 650
                    image_bytes = generate_certificate(last_year_student_name, "COMIC.TTF", "./2024_certificates/certificate for NMO.png", dict_nmo_2024, 650, 1150, 650, 710)
                    download_name = last_year_student_name
                elif certificate_type == "TST":
                    # Reduced x_symbol from 750 to 650
                    image_bytes = generate_certificate(tst_2024_student_name, "COMIC.TTF", "./2024_certificates/certificate for TST.png", dict_tst_2024, 650, 1150, 650, 710)
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
                # Reduced x_symbol from 750 to 650
                image_bytes = generate_certificate(student_name, "COMIC.TTF", "./for_certificates/certificate for DMO.png", dmo_dict, 600, 1100, 650, 710)
            elif certificate_type == "NMO":
                # Reduced x_symbol from 750 to 650
                image_bytes = generate_certificate(student_name, "COMIC.TTF", "./for_certificates/certificate for NMO.png", dictionary_for_certificates, 650, 1150, 650, 710)
            
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
                # Reduced x_symbol from 750 to 650
                image_bytes = generate_certificate(student_name, "COMIC.TTF", "./for_certificates/pretst2023.png", top100_dict, 750, 1150, 650, 710)
            elif certificate_type == "TST":
                # Reduced x_symbol from 750 to 650
                image_bytes = generate_certificate(student_name, "COMIC.TTF", "./for_certificates/TST round certificate.png", top25_dict, 750, 1150, 650, 710)

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
