import streamlit as st
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import os


def convert_to_dict(df):
    return dict(zip(df['Name'], df['Registration No.']))

# DATA ##


def load_data():
    data = pd.read_csv('./MAN/final/DMO_final_score.csv')
    nmo = pd.read_csv("./NMO_result/NMO-Result.csv")
    top25 = pd.read_csv("./NMO_result/top25.csv")
    top100 = pd.read_csv("./NMO_result/top100.csv")
    dmo_2024 = pd.read_csv('./2024 result/DMO-2024.csv')
    pmo_2024 = pd.read_csv('./2024 result/PMO-2024.csv')
    nmo_2024 = pd.read_csv('./2024 result/NMO-2024.csv')
    tst_2024 = pd.read_csv('./2024 result/TST-2024.csv')
    return data, nmo, top25, top100, dmo_2024, pmo_2024, nmo_2024, tst_2024


def create_dicts(data, nmo, top25, top100, dmo_2024, pmo_2024, nmo_2024, tst_2024):
    dmo_dict = convert_to_dict(data)
    dictionary_for_certificates = convert_to_dict(nmo)
    top25_dict = convert_to_dict(top25)
    top100_dict = convert_to_dict(top100)
    dict_dmo_2024 = convert_to_dict(dmo_2024)
    dict_pmo_2024 = convert_to_dict(pmo_2024)
    dict_nmo_2024 = convert_to_dict(nmo_2024)
    dict_tst_2024 = convert_to_dict(tst_2024)
    return dmo_dict, dictionary_for_certificates, top25_dict, top100_dict, dict_dmo_2024, dict_pmo_2024, dict_nmo_2024, dict_tst_2024


def generate_certificate(name, font_path, certificate_path, student_dict, x_name, y_name, x_symbol, y_symbol):
    name = name.title()
    if name in student_dict:
        image = Image.open(certificate_path)
        symbol_no = str(student_dict[name])
        draw = ImageDraw.Draw(image)
        font1 = ImageFont.truetype(font_path, 150)
        font2 = ImageFont.truetype(font_path, 70)
        draw.text((x_name, y_name), name, font=font1, fill=(0, 0, 0))
        draw.text((x_symbol, y_symbol), symbol_no, font=font2, fill=(0, 0, 0))
        image_bytes = io.BytesIO()
        image.save(image_bytes, 'PNG')
        image_bytes.seek(0)
        return image_bytes
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
    data, nmo, top25, top100, dmo_2024, pmo_2024, nmo_2024, tst_2024 = load_data()
    score = data['Score']
    dmo_dict, dictionary_for_certificates, top25_dict, top100_dict, dict_dmo_2024, dict_pmo_2024, dict_nmo_2024, dict_tst_2024 = create_dicts(data, nmo, top25, top100, dmo_2024, pmo_2024, nmo_2024, tst_2024)

    st.title("Student Certificate Generator and Stats Viewer for IMO 2023 and 2024\nBy Prajit Adhikari")

    menu = ["Generate Certificate for 2024 contests", "Home", "Generate Certificate for PreTST and TST for 2023 IMO", "View Statistics"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Generate Certificate for 2024 contests":
        st.subheader("Generate Certificate for 2024 IMO")
        student_name = st.selectbox("Select the name of the student (For DMO and PMO): ", sorted(dmo_2024['Name']))
        last_year_student_name = st.selectbox("Select the name for NMO: ", sorted(nmo_2024['Name']))
        tst_2024_student_name = st.selectbox("Select the name for TST: ", sorted(tst_2024['Name']))
        certificate_type = st.selectbox("Select certificate type", ["DMO", "PMO", "NMO", "TST"])
        if st.button("Generate"):
            if certificate_type == "DMO":
                image_bytes = generate_certificate(student_name, "COMIC.TTF", "./2024_certificates/certificate for DMO.png", dict_dmo_2024, 600, 1100, 750, 710)
            elif certificate_type == "PMO":
                image_bytes = generate_certificate(student_name, "COMIC.TTF", "./2024_certificates/certificate for PMO.png", dict_pmo_2024, 650, 1150, 750, 710)
            elif certificate_type == "NMO":
                image_bytes = generate_certificate(last_year_student_name, "COMIC.TTF", "./2024_certificates/certificate for NMO.png", dict_nmo_2024, 650, 1150, 750, 710)
            elif certificate_type == "TST":
                image_bytes = generate_certificate(tst_2024_student_name, "COMIC.TTF", "./2024_certificates/certificate for TST.png", dict_tst_2024, 650, 1150, 750, 710)

            if image_bytes is not None:
                st.image(image_bytes, caption='Generated certificate')
                download_name = student_name if certificate_type != "NMO" else last_year_student_name if certificate_type == "NMO" else tst_2024_student_name
                st.download_button(
                    "Download Certificate",
                    data=image_bytes,
                    file_name=f'{download_name}_{certificate_type}_2024_certificate.png',
                    mime='image/png'
                )

    elif choice == "Home":
        st.subheader("Home")
        st.write("Welcome to the Student Certificate Generator and Stats Viewer. Please select an action from the sidebar. "
                 "Please find your registration/symbol number here.")
        st.subheader("Generate Certificate")
        student_name = st.selectbox("Select the name of the student: ", sorted(data['Name of Students'].unique()))
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
        student_name = st.selectbox("Select the name of the student: ", sorted(top100['Name of Students']))
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
        if st.button("Show Distribution"):
            plot_distribution(score)
        if st.button("Show Stats"):
            show_stats(score)


if __name__ == "__main__":
    main()
