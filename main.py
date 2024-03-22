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

## DATA ##
data = pd.read_csv('./MAN/final/DMO_final_score.csv')
score = data['Score']
sorted_score = sorted(score)
# igo_data = pd.read_csv("./IGO/IGO scores.csv")
nmo = pd.read_csv("./NMO_result/NMO-Result.csv")
dictionary_for_certificates = dict(zip(nmo['Name of Students'], nmo['Registration No.']))
top25 = pd.read_csv("./NMO_result/top25.csv")
top100 = pd.read_csv("./NMO_result/top100.csv")
dmo_dict = dict(zip(data['Name of Students'], data['Registration No.']))
top25_dict = dict(zip(top25['Name of Students'], data['Registration No.']))
top100_dict = dict(zip(top100['Name of Students'], data['Registration No.']))
dict_for_pretst = dict(zip(top100['Name of Students'], top100['Registration No.']))


## 2024 data
dmo_2024 = pd.read_csv('./2024 result/DMO-2024.csv')
pmo_2024 = pd.read_csv('./2024 result/PMO-2024.csv')
nmo_2024 = pd.read_csv('./2024 result/NMO-2024.csv')

dict_dmo_2024 = convert_to_dict(dmo_2024)
dict_pmo_2024 = convert_to_dict(pmo_2024)
dict_nmo_2024 = convert_to_dict(nmo_2024)

print(dict_dmo_2024)

## FUNCTIONS ##
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
    if name in top100_dict['Name of Students']:
        symbol_no = str(dict_for_pretst[name])
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

def generate_pmo_certificate(name, font_path, certificate_path):
    name = name.title()
    if name in dict_pmo_2024:
        symbol_no = str(dict_pmo_2024[name])
        image = Image.open(certificate_path)
        draw = ImageDraw.Draw(image)
        font1 = ImageFont.truetype(font_path, 150)
        font2 = ImageFont.truetype(font_path, 70)
        draw.text((650, 710), symbol_no, font=font2, fill=(0, 0, 0))
        draw.text((690, 1150), name, font=font1, fill=(0, 0, 0))
        image_bytes = io.BytesIO()
        image.save(image_bytes, 'PNG')
        image_bytes.seek(0)
        return image_bytes
    else:
        st.error(f"{name} is not in the list of students.")
        return None

def generate_dmo2024_certificate(name, font_path, certificate_path):
    name = name.title()
    if name in dict_dmo_2024:
        symbol_no = str(dict_dmo_2024[name])
        image = Image.open(certificate_path)
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

def generate_nmo2024_certificate(name, font_path, certificate_path):
    name = name.title()
    if name in dict_nmo_2024:
        symbol_no = str(dict_nmo_2024[name])
        image = Image.open(certificate_path)
        draw = ImageDraw.Draw(image)
        font1 = ImageFont.truetype(font_path, 150)
        font2 = ImageFont.truetype(font_path, 70)
        draw.text((650, 710), symbol_no, font=font2, fill=(0,0,0))
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
        symbol_no = str(dict_for_pretst[name])
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
    image_bytes = io.BytesIO()
    st.title("Student Certificate Generator and Stats Viewer for IMO 2023 and 2024")

    menu = [  "Generate Certificate for 2024 contests","Home",
              "Generate Certificate for PreTST and TST", "View Statistics"]
    choice = st.sidebar.selectbox("Menu", menu)


    if choice == "Generate Certificate for 2024 contests":
        st.subheader("Generate Certificate for 2024")
        st.write("Please select the name of the student and the type of certificate you want to generate. To generate "
                 "a certificate for NMO, you need to select the name of the student who participated in the last year's NMO."
                 "Please note that the name of the student should be the same as the one in the list. Also, for DMO and "
                 "PMO, select the name of the student from the list from the DMO and PMO name list.")
        student_name = st.selectbox("Select the name of the student(For DMO and PMO): ", sorted(dmo_2024['Name']))
        last_year_student_name = st.selectbox("Select the name for NMO: ", sorted(nmo_2024['Name']))
        certificate_type = st.selectbox("Select certificate type", ["DMO", "PMO", "NMO"])
        if st.button("Generate"):
            if certificate_type == "DMO":
                image_bytes = generate_dmo2024_certificate(student_name, "COMIC.TTF",
                                                   "./2024_certificates/certificate for DMO.png")
            elif certificate_type == "PMO":
                if student_name in dict_pmo_2024:
                    image_bytes = generate_pmo_certificate(student_name, "COMIC.TTF",
                                                   "./2024_certificates/certificate for PMO.png")
                else:
                    st.error(f"{student_name} is not in the list of students.")
                    return None
            elif certificate_type == "NMO":
                if student_name in dict_nmo_2024 or last_year_student_name in dict_nmo_2024:
                    image_bytes = generate_nmo2024_certificate(last_year_student_name, "COMIC.TTF",
                                                       "./2024_certificates/certificate for NMO.png")
                else:
                    st.error(f"{student_name} is not in the list of students.")
                    return None

            if image_bytes is not None:
                st.image(image_bytes, caption='Generated certificate')
                st.download_button(
                    "Download Certificate",
                    data=image_bytes,
                    file_name=f'{student_name}_certificate.png',
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
