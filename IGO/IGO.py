from PIL import Image, ImageDraw, ImageFont
import csv



def write_text_to_certificate(name):
    # Open the background image
    certificate_template = Image.open("IGO certificate.png")
    draw = ImageDraw.Draw(certificate_template)
    font = ImageFont.truetype("arial.ttf", 36)

    # Write the name to the certificate
    (x, y) = (450, 350)
    draw.text((x, y), name, fill=(0, 0, 0), font=font)

    # Save the output image
    certificate_template.save("certificate_with_text.png")


# Example usage:
with open('IGO scores.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        name = row[0]
        write_text_to_certificate(name)
