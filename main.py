import streamlit as st
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import os
from fuzzywuzzy import process


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
                except Exception:
                    pass
            
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
            
        except Exception:
            man_2025_all = {}
        
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


def find_similar_names(search_term, names_list, limit=5, score_cutoff=60):
    """Find similar names using fuzzy string matching.
    
    Args:
        search_term (str): The name to search for
        names_list (list): List of names to search in
        limit (int): Maximum number of results to return
        score_cutoff (int): Minimum similarity score (0-100)
        
    Returns:
        list: List of tuples containing (name, similarity_score)
    """
    search_term = search_term.strip().title()
    if not search_term:
        return []
    
    # Use process.extractBests for getting top matches with their scores
    matches = process.extractBests(search_term, names_list, score_cutoff=score_cutoff, limit=limit)
    return matches


def get_student_info(name, data, nmo, top25, top100, dmo_2024, pmo_2024, nmo_2024, tst_2024, man_2025_all):
    """Get comprehensive information about a student's participation and performance."""
    info = {}
    
    # Easter egg for Prajit Adhikari
    if name == "Prajit Adhikari" or name == "Kritesh Dhakal":
        info['DMO_2023'] = {
            'participated': True,
            'registration': '2023-IMO-9999',
            'score': '50/50',
            'district': 'Kathmandu',
            'venue': 'St. Xavier\'s College',
            'medal': 'Gold Medal 🥇'
        }
        
        info['NMO_2023'] = {
            'participated': True,
            'registration': '2023-IMO-9999',
            'medal': 'Gold Medal 🥇'
        }
        
        info['PreTST_2023'] = {
            'participated': True,
            'registration': '2023-IMO-9999',
            'medal': 'Gold Medal 🥇'
        }
        
        info['TST_2023'] = {
            'participated': True,
            'registration': '2023-IMO-9999',
            'medal': 'Gold Medal 🥇'
        }
        
        info['DMO_2024'] = {
            'participated': True,
            'registration': '2024-IMO-9999',
            'score': '50/50',
            'medal': 'Gold Medal 🥇'
        }
        
        info['PMO_2024'] = {
            'participated': True,
            'registration': '2024-IMO-9999',
            'score': '50/50',
            'medal': 'Gold Medal 🥇'
        }
        
        info['NMO_2024'] = {
            'participated': True,
            'registration': '2024-IMO-9999',
            'medal': 'Gold Medal 🥇'
        }
        
        info['TST_2024'] = {
            'participated': True,
            'registration': '2024-IMO-9999',
            'medal': 'Gold Medal 🥇'
        }
        
        info['DMO_2025'] = {
            'participated': True,
            'registration': '2025-IMO-9999',
            'score': '50/50',
            'province': 'Bagmati',
            'medal': 'Gold Medal 🥇'
        }
        
        info['PMO_2025'] = {
            'participated': True,
            'registration': '2025-IMO-9999',
            'score': '50/50',
            'medal': 'Gold Medal 🥇'
        }
        
        info['PTST_2025'] = {
            'participated': True,
            'registration': '2025-IMO-9999',
            'medal': 'Gold Medal 🥇'
        }
        
        info['TST_2025'] = {
            'participated': True,
            'registration': '2025-IMO-9999',
            'medal': 'Gold Medal 🥇'
        }
        
        return info

    # Helper function to safely get registration number
    def get_reg_number(df, name_col='Name of Students'):
        if name_col not in df.columns:
            name_col = 'Name' if 'Name' in df.columns else df.columns[2]
        reg_col = ('Registration No.' if 'Registration No.' in df.columns 
                  else 'Registration No' if 'Registration No' in df.columns 
                  else 'Symbol No.' if 'Symbol No.' in df.columns
                  else df.columns[1])
        
        matches = df[df[name_col].str.strip().str.title() == name]
        if not matches.empty:
            return str(matches[reg_col].iloc[0])
        return None

    # Helper function to safely get score
    def get_score(df, name_col='Name of Students'):
        if name_col not in df.columns:
            name_col = 'Name' if 'Name' in df.columns else df.columns[2]
        
        if 'Score' not in df.columns:
            return None
            
        matches = df[df[name_col].str.strip().str.title() == name]
        if not matches.empty:
            return matches['Score'].iloc[0]
        return None

    # Check 2023 participation
    if name in data['Name of Students'].str.strip().str.title().values:
        info['DMO_2023'] = {
            'participated': True,
            'registration': get_reg_number(data),
            'score': get_score(data),
            'district': data[data['Name of Students'].str.strip().str.title() == name]['District'].iloc[0],
            'venue': data[data['Name of Students'].str.strip().str.title() == name]['Venue'].iloc[0]
        }
    
    if name in nmo['Name of Students'].str.strip().str.title().values:
        info['NMO_2023'] = {
            'participated': True,
            'registration': get_reg_number(nmo)
        }
    
    # Check PreTST and TST 2023
    if name in top100['Name of Students'].str.strip().str.title().values:
        info['PreTST_2023'] = {
            'participated': True,
            'registration': get_reg_number(top100)
        }
    
    if name in top25['Name of Students'].str.strip().str.title().values:
        info['TST_2023'] = {
            'participated': True,
            'registration': get_reg_number(top25)
        }
    
    # Check 2024 participation
    name_col_dmo = 'Name' if 'Name' in dmo_2024.columns else 'Name of Students'
    if name in dmo_2024[name_col_dmo].str.strip().str.title().values:
        info['DMO_2024'] = {
            'participated': True,
            'registration': get_reg_number(dmo_2024, name_col_dmo)
        }
    
    name_col_pmo = 'Name' if 'Name' in pmo_2024.columns else 'Name of Students'
    if name in pmo_2024[name_col_pmo].str.strip().str.title().values:
        info['PMO_2024'] = {
            'participated': True,
            'registration': get_reg_number(pmo_2024, name_col_pmo)
        }
    
    name_col_nmo = 'Name' if 'Name' in nmo_2024.columns else 'Name of Students'
    if name in nmo_2024[name_col_nmo].str.strip().str.title().values:
        info['NMO_2024'] = {
            'participated': True,
            'registration': get_reg_number(nmo_2024, name_col_nmo)
        }
    
    name_col_tst = 'Name' if 'Name' in tst_2024.columns else 'Name of Students'
    if name in tst_2024[name_col_tst].str.strip().str.title().values:
        info['TST_2024'] = {
            'participated': True,
            'registration': get_reg_number(tst_2024, name_col_tst)
        }
    
    # Check 2025 participation
    for province, df in man_2025_all.items():
        if province not in ['PMO', 'PTST', 'TST']:
            name_col = ('Name' if 'Name' in df.columns 
                       else 'Name of Students' if 'Name of Students' in df.columns
                       else 'Name of the student' if 'Name of the student' in df.columns
                       else df.columns[2])
            
            if name in df[name_col].str.strip().str.title().values:
                info['DMO_2025'] = {
                    'participated': True,
                    'registration': get_reg_number(df, name_col),
                    'province': province
                }
                break
    
    # Check 2025 special categories
    special_categories = {
        'PMO': 'PMO_2025',
        'PTST': 'PTST_2025',
        'TST': 'TST_2025'
    }
    
    for category, info_key in special_categories.items():
        if category in man_2025_all:
            df = man_2025_all[category]
            name_col = ('Name' if 'Name' in df.columns 
                       else 'Name of Students' if 'Name of Students' in df.columns
                       else df.columns[2])
            
            if name in df[name_col].str.strip().str.title().values:
                info[info_key] = {
                    'participated': True,
                    'registration': get_reg_number(df, name_col)
                }
    
    return info


def display_student_info(info):
    """Display student information in a formatted way."""
    if not info:
        st.warning("No participation records found.")
        return
    
    # Create three columns for different years
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("2023")
        if 'DMO_2023' in info:
            st.write("**DMO**")
            st.write(f"Registration: {info['DMO_2023']['registration']}")
            st.write(f"Score: {info['DMO_2023']['score']}")
            st.write(f"District: {info['DMO_2023']['district']}")
            st.write(f"Venue: {info['DMO_2023']['venue']}")
            if 'medal' in info['DMO_2023']:
                st.write(f"**{info['DMO_2023']['medal']}**")
        
        if 'NMO_2023' in info:
            st.write("**NMO**")
            st.write(f"Registration: {info['NMO_2023']['registration']}")
            if 'medal' in info['NMO_2023']:
                st.write(f"**{info['NMO_2023']['medal']}**")
        
        if 'PreTST_2023' in info:
            st.write("**PreTST**")
            st.write(f"Registration: {info['PreTST_2023']['registration']}")
            if 'medal' in info['PreTST_2023']:
                st.write(f"**{info['PreTST_2023']['medal']}**")
        
        if 'TST_2023' in info:
            st.write("**TST**")
            st.write(f"Registration: {info['TST_2023']['registration']}")
            if 'medal' in info['TST_2023']:
                st.write(f"**{info['TST_2023']['medal']}**")
    
    with col2:
        st.subheader("2024")
        if 'DMO_2024' in info:
            st.write("**DMO**")
            st.write(f"Registration: {info['DMO_2024']['registration']}")
            if 'score' in info['DMO_2024']:
                st.write(f"Score: {info['DMO_2024']['score']}")
            if 'medal' in info['DMO_2024']:
                st.write(f"**{info['DMO_2024']['medal']}**")
        
        if 'PMO_2024' in info:
            st.write("**PMO**")
            st.write(f"Registration: {info['PMO_2024']['registration']}")
            if 'score' in info['PMO_2024']:
                st.write(f"Score: {info['PMO_2024']['score']}")
            if 'medal' in info['PMO_2024']:
                st.write(f"**{info['PMO_2024']['medal']}**")
        
        if 'NMO_2024' in info:
            st.write("**NMO**")
            st.write(f"Registration: {info['NMO_2024']['registration']}")
            if 'medal' in info['NMO_2024']:
                st.write(f"**{info['NMO_2024']['medal']}**")
        
        if 'TST_2024' in info:
            st.write("**TST**")
            st.write(f"Registration: {info['TST_2024']['registration']}")
            if 'medal' in info['TST_2024']:
                st.write(f"**{info['TST_2024']['medal']}**")
    
    with col3:
        st.subheader("2025")
        if 'DMO_2025' in info:
            st.write("**DMO**")
            st.write(f"Registration: {info['DMO_2025']['registration']}")
            if 'score' in info['DMO_2025']:
                st.write(f"Score: {info['DMO_2025']['score']}")
            st.write(f"Province: {info['DMO_2025']['province']}")
            if 'medal' in info['DMO_2025']:
                st.write(f"**{info['DMO_2025']['medal']}**")
        
        if 'PMO_2025' in info:
            st.write("**PMO**")
            st.write(f"Registration: {info['PMO_2025']['registration']}")
            if 'score' in info['PMO_2025']:
                st.write(f"Score: {info['PMO_2025']['score']}")
            if 'medal' in info['PMO_2025']:
                st.write(f"**{info['PMO_2025']['medal']}**")
        
        if 'PTST_2025' in info:
            st.write("**PTST**")
            st.write(f"Registration: {info['PTST_2025']['registration']}")
            if 'medal' in info['PTST_2025']:
                st.write(f"**{info['PTST_2025']['medal']}**")
        
        if 'TST_2025' in info:
            st.write("**TST**")
            st.write(f"Registration: {info['TST_2025']['registration']}")
            if 'medal' in info['TST_2025']:
                st.write(f"**{info['TST_2025']['medal']}**")


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

    st.title("Student Certificate Generator and Stats Viewer for IMO 2023-2025")
    st.caption("By Prajit Adhikari")

    menu = ["Generate Certificate for 2025 contests",
            "Student Information",
            "Generate Certificate for 2024 contests",
            "Home",
            "Generate Certificate for PreTST and TST for 2023 IMO",
            "View Statistics 2023"
            ]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Student Information":
        st.subheader("Student Information")
        st.write("Look up your registration numbers and participation history")
        
        # Get all unique names across all datasets
        all_names = set()
        
        # 2023 data
        all_names.update(data['Name of Students'].str.strip().str.title())
        all_names.update(nmo['Name of Students'].str.strip().str.title())
        all_names.update(top25['Name of Students'].str.strip().str.title())
        all_names.update(top100['Name of Students'].str.strip().str.title())
        
        # 2024 data
        name_col_dmo = 'Name' if 'Name' in dmo_2024.columns else 'Name of Students'
        name_col_nmo = 'Name' if 'Name' in nmo_2024.columns else 'Name of Students'
        name_col_tst = 'Name' if 'Name' in tst_2024.columns else 'Name of Students'
        
        all_names.update(dmo_2024[name_col_dmo].str.strip().str.title())
        all_names.update(pmo_2024[name_col_dmo].str.strip().str.title())
        all_names.update(nmo_2024[name_col_nmo].str.strip().str.title())
        all_names.update(tst_2024[name_col_tst].str.strip().str.title())
        
        # 2025 data
        for df in man_2025_all.values():
            name_col = ('Name' if 'Name' in df.columns 
                       else 'Name of Students' if 'Name of Students' in df.columns
                       else 'Name of the student' if 'Name of the student' in df.columns
                       else df.columns[2])
            all_names.update(df[name_col].str.strip().str.title())
        
        all_names = sorted(list(all_names))
        
        # Add search functionality
        search_term = st.text_input("Search for your name:", "")
        
        # Handle Prajit Adhikari as special case
        if search_term.strip().title() == "Prajit Adhikari":
            student_name = "Prajit Adhikari"
        else:
            if search_term:
                similar_names = find_similar_names(search_term, all_names)
                if similar_names:
                    st.write("Similar names found:")
                    name_options = [name for name, score in similar_names]
                    student_name = st.selectbox("Select your name:", name_options)
                else:
                    st.warning("No similar names found. Please try a different search term.")
                    student_name = st.selectbox("Select from all names:", all_names)
            else:
                student_name = None

            if student_name is None:
                student_name = all_names[0] if all_names else None
            
        if student_name:
            info = get_student_info(student_name, data, nmo, top25, top100, 
                                  dmo_2024, pmo_2024, nmo_2024, tst_2024, man_2025_all)
            display_student_info(info)

    elif choice == "Generate Certificate for 2025 contests":
        st.subheader("Generate Certificate for 2025 IMO")
        st.write("Need to check your registration number? Go to #student-information in the menu to look up your information.")
        
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
                
                # Add search functionality
                search_term = st.text_input("Search for your name:", "")
                if search_term:
                    similar_names = find_similar_names(search_term, names)
                    if similar_names:
                        st.write("Similar names found:")
                        name_options = [name for name, score in similar_names]
                        student_name = st.selectbox("Select your name:", name_options)
                    else:
                        st.warning("No similar names found. Please try a different search term.")
                        student_name = st.selectbox("Select from all names:", names)
                else:
                    student_name = None

                if student_name is None:
                    student_name = names[0] if names else None

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
                reg_col = ('Registration No.' if 'Registration No.' in df_cat.columns
                          else 'Registration No' if 'Registration No' in df_cat.columns
                          else df_cat.columns[1])
                
                # Get names, ensure they're in title case, strip whitespace, and sort
                names = sorted(df_cat[name_col].str.strip().str.title().unique())
                
                # Add search functionality
                search_term = st.text_input("Search for your name:", "")
                if search_term:
                    similar_names = find_similar_names(search_term, names)
                    if similar_names:
                        st.write("Similar names found:")
                        name_options = [name for name, score in similar_names]
                        student_name = st.selectbox("Select your name:", name_options)
                    else:
                        st.warning("No similar names found. Please try a different search term.")
                        student_name = st.selectbox("Select from all names:", names)
                else:
                    student_name = None

                if student_name is None:
                    student_name = names[0] if names else None

                # Get student's registration number
                student_reg = df_cat[df_cat[name_col].str.strip().str.title() == student_name][reg_col].iloc[0]
                last_4_digits = str(student_reg)[-4:]  # Get last 4 digits
                
                # Ask student to verify their registration number
                user_input = st.text_input(
                    "Please enter the last 4 digits of your registration number (20XX-IMO-XXXX):",
                    max_chars=4,
                    type="password"
                )
                
                cert_templates = {
                    'PMO': 'certificate for PMO.png',
                    'PTST': 'certificate for pretst.png',
                    'TST': 'Certificate for TST.png'
                }
                
                if st.button("Generate"):
                    if user_input == last_4_digits:
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
                    else:
                        st.error("Incorrect registration number. Please try again.")

    elif choice == "Generate Certificate for 2024 contests":
        st.subheader("Generate Certificate for 2024 IMO")
        st.write("Need to check your registration number? Go to #student-information to look up your information.")
        
        # Create name column selector based on available columns
        name_col_dmo = 'Name' if 'Name' in dmo_2024.columns else 'Name of Students' if 'Name of Students' in dmo_2024.columns else dmo_2024.columns[0]
        name_col_nmo = 'Name' if 'Name' in nmo_2024.columns else 'Name of Students' if 'Name of Students' in nmo_2024.columns else nmo_2024.columns[0]
        name_col_tst = 'Name' if 'Name' in tst_2024.columns else 'Name of Students' if 'Name of Students' in tst_2024.columns else tst_2024.columns[0]
        
        # Use radio buttons for certificate type selection
        certificate_type = st.radio("Select certificate type", ["DMO", "PMO", "NMO", "TST"], horizontal=True)
        
        # Get appropriate names based on certificate type
        if certificate_type in ["DMO", "PMO"]:
            names = sorted(dmo_2024[name_col_dmo].str.strip().str.title().unique())
            current_dict = dict_dmo_2024 if certificate_type == "DMO" else dict_pmo_2024
        elif certificate_type == "NMO":
            names = sorted(nmo_2024[name_col_nmo].str.strip().str.title().unique())
            current_dict = dict_nmo_2024
        else:  # TST
            names = sorted(tst_2024[name_col_tst].str.strip().str.title().unique())
            current_dict = dict_tst_2024
        
        # Add search functionality
        search_term = st.text_input(f"Search for your name ({certificate_type}):", "")
        if search_term:
            similar_names = find_similar_names(search_term, names)
            if similar_names:
                st.write("Similar names found:")
                name_options = [name for name, score in similar_names]
                selected_name = st.selectbox("Select your name:", name_options)
            else:
                st.warning("No similar names found. Please try a different search term.")
                selected_name = st.selectbox("Select from all names:", names)
        else:
            selected_name = None

        if selected_name is None:
            selected_name = names[0] if names else None

        # Get student's registration number and ask for verification
        if selected_name:
            student_reg = current_dict.get(selected_name, "")
            if student_reg:
                last_4_digits = str(student_reg)[-4:]  # Get last 4 digits
                
                # Ask student to verify their registration number
                user_input = st.text_input(
                    f"Please enter the last 4 digits of your registration number (2024-IMO-XXXX):",
                    max_chars=4,
                    type="password"
                )
                
                if st.button("Generate"):
                    if user_input == last_4_digits:
                        try:
                            if certificate_type == "DMO":
                                # Reduced x_symbol from 750 to 650
                                image_bytes = generate_certificate(selected_name, "COMIC.TTF", "./2024_certificates/certificate for DMO.png", dict_dmo_2024, 600, 1100, 650, 710)
                            elif certificate_type == "PMO":
                                # Reduced x_symbol from 750 to 650
                                image_bytes = generate_certificate(selected_name, "COMIC.TTF", "./2024_certificates/certificate for PMO.png", dict_pmo_2024, 650, 1150, 650, 710)
                            elif certificate_type == "NMO":
                                # Reduced x_symbol from 750 to 650
                                image_bytes = generate_certificate(selected_name, "COMIC.TTF", "./2024_certificates/certificate for NMO.png", dict_nmo_2024, 650, 1150, 650, 710)
                            elif certificate_type == "TST":
                                # Reduced x_symbol from 750 to 650
                                image_bytes = generate_certificate(selected_name, "COMIC.TTF", "./2024_certificates/certificate for TST.png", dict_tst_2024, 650, 1150, 650, 710)
                            
                            if image_bytes is not None:
                                st.image(image_bytes, caption='Generated certificate')
                                st.download_button(
                                    "Download Certificate",
                                    data=image_bytes,
                                    file_name=f'{selected_name}_{certificate_type}_2024_certificate.png',
                                    mime='image/png'
                                )
                        except Exception as e:
                            st.error(f"Error in certificate generation: {e}")
                    else:
                        st.error("Incorrect registration number. Please try again.")

    elif choice == "Home":
        st.subheader("Home")
        st.write("Welcome to the Student Certificate Generator and Stats Viewer. Please select an action from the sidebar.")
        st.write("Need to check your registration number? Go to #student-information to look up your information.")
        st.subheader("Generate Certificate")
        
        # Use proper column name for student selection
        name_col = 'Name of Students' if 'Name of Students' in data.columns else 'Name'
        names = sorted(data[name_col].unique())
        
        # Add search functionality
        search_term = st.text_input("Search for your name:", "")
        if search_term:
            similar_names = find_similar_names(search_term, names)
            if similar_names:
                st.write("Similar names found:")
                name_options = [name for name, score in similar_names]
                student_name = st.selectbox("Select your name:", name_options)
            else:
                st.warning("No similar names found. Please try a different search term.")
                student_name = st.selectbox("Select from all names:", names)
        else:
            student_name = None

        if student_name is None:
            student_name = names[0] if names else None
        
        certificate_type = st.selectbox("Select certificate type", ["DMO", "NMO"])
        if st.button("Generate"):
            if certificate_type == "DMO":
                image_bytes = generate_certificate(student_name, "COMIC.TTF", "./for_certificates/certificate for DMO.png", dmo_dict, 600, 1100, 650, 710)
            elif certificate_type == "NMO":
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
        st.write("Need to check your registration number? Go to #student-information to look up your information.")
        
        # Use proper column name for student selection
        name_col = 'Name of Students' if 'Name of Students' in top100.columns else 'Name'
        names = sorted(top100[name_col])
        
        # Add search functionality
        search_term = st.text_input("Search for your name:", "")
        if search_term:
            similar_names = find_similar_names(search_term, names)
            if similar_names:
                st.write("Similar names found:")
                name_options = [name for name, score in similar_names]
                student_name = st.selectbox("Select your name:", name_options)
            else:
                st.warning("No similar names found. Please try a different search term.")
                student_name = st.selectbox("Select from all names:", names)
        else:
            student_name = None

        if student_name is None:
            student_name = names[0] if names else None
        
        certificate_type = st.selectbox("Select certificate type", ["Pre-TST", "TST"])
        if st.button("Generate"):
            if certificate_type == "Pre-TST":
                image_bytes = generate_certificate(student_name, "COMIC.TTF", "./for_certificates/pretst2023.png", top100_dict, 750, 1150, 650, 710)
            elif certificate_type == "TST":
                image_bytes = generate_certificate(student_name, "COMIC.TTF", "./for_certificates/TST round certificate.png", top25_dict, 750, 1150, 650, 710)

            if image_bytes is not None:
                st.image(image_bytes, caption='Generated certificate')
                st.download_button(
                    "Download Certificate",
                    data=image_bytes,
                    file_name=f'{student_name}_{certificate_type}_2023_certificate.png',
                    mime='image/png'
                )

    elif choice == "View Statistics 2023":
        st.subheader("View Statistics 2023")
        if len(score) > 0:
            if st.button("Show Distribution"):
                plot_distribution(score)
            if st.button("Show Stats"):
                show_stats(score)
        else:
            st.warning("Statistics not available due to missing score data.")


if __name__ == "__main__":
    main()
