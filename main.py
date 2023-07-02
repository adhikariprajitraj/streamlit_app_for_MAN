import streamlit as st
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import os


## DATA ##
data = pd.read_csv('./MAN/final/DMO_final_score.csv')
score = data['Score']
sorted_score = sorted(score)
igo_data = pd.read_csv("./IGO/IGO scores.csv")
nmo = pd.read_csv("./NMO_result/NMO-Result.csv")
dictionary_for_certificates = dict(zip(nmo['Name of Students'], nmo['Registration No.']))
top25 = pd.read_csv("./NMO_result/top25.csv")
top100 = pd.read_csv("./NMO_result/top100.csv")
dmo_dict = dict(zip(data['Name of Students'], data['Registration No.']))
top25_dict = dict(zip(top25['Name of Students'], data['Registration No.']))
top100_dict = dict(zip(top100['Name of Students'], data['Registration No.']))

def generate_certificate(name, font_path, certificate_path):
    name = name.title()
    if name in dmo_dict:
        image = Image.open(certificate_path)
        symbol_no = str(dmo_dict[name])
        draw = ImageDraw.Draw(image)
        font1 = ImageFont.truetype(font_path, 150)
        font2 = ImageFont.truetype(font_path, 70)
        draw.text((600, 1100), name, font=font1, fill=(0, 0, 0))
        draw.text((750, 710), symbol_no, font=font2, fill=(0, 0, 0))
        image_bytes = io.BytesIO()
        image.save(image_bytes, 'PNG')
        image_bytes.seek(0)
        return image_bytes
    else:
        st.error(f"{name} is not in the list of students.")
        return None


def plot_distribution():
    sns.histplot(score, stat='density', color='#44eecc')
    plt.savefig(fname='plot')
    st.image('plot.png')
    os.remove('plot.png')


def generate_nmo_certificate(name, font_path, certificate_path):
    name = name.title()
    if name in dictionary_for_certificates:
        symbol_no = str(dictionary_for_certificates[name])
        image = Image.open(certificate_path)
        draw = ImageDraw.Draw(image)
        font1 = ImageFont.truetype(font_path, 150)
        font2 = ImageFont.truetype(font_path, 70)
        draw.text((750, 710), symbol_no, font=font2, fill=(0,0,0))
        draw.text((690, 1150), name, font=font1, fill=(0,0,0))
        image_bytes = io.BytesIO()
        image.save(image_bytes, 'PNG')
        image_bytes.seek(0)
        return image_bytes
    else:
        st.error(f"{name} is not in the list of students.")
        return None


def generate_top100_certificate(name, font_path, certificate_path):
    name = name.title()
    if name in top100_dict:
        symbol_no = str(dictionary_for_certificates[name])
        image = Image.open(certificate_path)
        draw = ImageDraw.Draw(image)
        font1 = ImageFont.truetype(font_path, 150)
        font2 = ImageFont.truetype(font_path, 70)
        draw.text((750, 710), symbol_no, font=font2, fill=(0,0,0))
        draw.text((690, 1150), name, font=font1, fill=(0,0,0))
        image_bytes = io.BytesIO()
        image.save(image_bytes, 'PNG')
        image_bytes.seek(0)
        return image_bytes
    else:
        st.error(f"{name} is not in the list of students.")
        return None

def generate_top25_certificate(name, font_path, certificate_path):
    name = name.title()
    if name in top25_dict:
        symbol_no = str(dictionary_for_certificates[name])
        image = Image.open(certificate_path)
        draw = ImageDraw.Draw(image)
        font1 = ImageFont.truetype(font_path, 150)
        font2 = ImageFont.truetype(font_path, 70)
        draw.text((750, 710), symbol_no, font=font2, fill=(0,0,0))
        draw.text((690, 1150), name, font=font1, fill=(0,0,0))
        image_bytes = io.BytesIO()
        image.save(image_bytes, 'PNG')
        image_bytes.seek(0)
        return image_bytes
    else:
        st.error(f"{name} is not in the list of students.")
        return None

def show_stats():
    st.write(f'**Mean:**  {np.mean(score)}\n**Median:**  {np.median(score)}\n'
                   f'**Standard Deviation:  **{np.std(score)}\n**Variance:**  {np.var(score)}\n'
                   f'**Max: ** {np.max(score)} \n **Min:** {np.min(score)}')

def main():
    st.title("Student Certificate Generator and Stats Viewer")

    menu = [ "Generate Certificate", "Generate Certificate for PreTST and TST", "Home", "View Statistics"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.write("Welcome to the Student Certificate Generator and Stats Viewer. Please select an action from the sidebar.")
    elif choice == "Generate Certificate":
        st.subheader("Generate Certificate")
        student_name = st.selectbox("Select the name of the student: ", sorted(data['Name of Students'].unique()))

        certificate_type = st.selectbox("Select certificate type", ["DMO", "NMO"])
        if st.button("Generate"):
            if certificate_type == "DMO":
                image_bytes = generate_certificate(student_name, "COMIC.TTF",
                                                   "./for_certificates/certificate for DMO.png")
            elif certificate_type == "NMO":
                image_bytes = generate_nmo_certificate(student_name, "COMIC.TTF",
                                                       "./for_certificates/certificate for NMO.png")


            if image_bytes is not None:
                st.image(image_bytes, caption='Generated certificate')
                st.download_button(
                    "Download Certificate",
                    data=image_bytes,
                    file_name=f'{student_name}_certificate.png',
                    mime='image/png'
                )

    elif choice == "Generate Certificate for PreTST and TST":
        st.subheader("Generate Certificate for PreTST and TST")
        student_name = st.selectbox("Select the name of the student: ", sorted(top100['Name of Students']))
        certificate_type = st.selectbox("Select certificate type", ["Pre-TST", "TST"])
        if st.button("Generate"):
            if certificate_type == "Pre-TST":
                image_bytes = generate_top100_certificate(student_name, "COMIC.TTF",
                                                  "./for_certificates/certificate for pretst.png")
            elif certificate_type == "TST":
                image_bytes = generate_top25_certificate(student_name, "COMIC.TTF",
                                                 "./for_certificates/TST round certificate.png")

        if image_bytes is not None:
            st.image(image_bytes, caption='Generated certificate')
            st.download_button(
                "Download Certificate",
                data=image_bytes,
                file_name=f'{student_name}_certificate.png',
                mime='image/png'
            )
    elif choice == "View Statistics":
        st.subheader("View Statistics")
        if st.button("Show Distribution"):
            plot_distribution()
        if st.button("Show Stats"):
            show_stats()


if __name__ == "__main__":
    main()
