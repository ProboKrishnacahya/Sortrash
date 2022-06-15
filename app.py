import PySimpleGUI as sg
import os
from sortrash import predictMain
import io
from PIL import Image

working_directory = os.getcwd()
font = ('Cascadia Mono', 14)
sg.set_options(font=font)

layout = [  
            [sg.Text("Archotech 2022 - Simplify the Trash Sorting")],
            [sg.Text("*******************************************")],
            [sg.Text("Unggah Gambar Sampah:")],
            [sg.InputText(size=(64, 2), key="-FILE_PATH-"), 
            sg.FileBrowse(initial_folder=working_directory, file_types=[("Image Files", "*.jpg")])],
            [sg.Button('Submit'), sg.Exit()]
        ]

window = sg.Window("Sortrash", layout, resizable=True).Finalize()
window.Maximize()

def predict(csv_address):
    file = open(csv_address)
    pil_image = Image.open(file.name)
    pil_image.thumbnail((400, 400))
    png_bio = io.BytesIO()
    pil_image.save(png_bio, format="PNG")
    png_data = png_bio.getvalue()
    resultLabel, descriptionLabel, precision, recall, accuracy, f1 =  predictMain(file.name)
    layout2 = [
            [sg.Image(data=png_data)],
            [sg.Text('Tuas bergerak membuka Tong Sampah '+resultLabel+'.')],
            [sg.Text('Sampah '+resultLabel+' adalah '+descriptionLabel+'.')],
            [sg.Text('\n==================================================\n')],
            [sg.Text('Precision: '+precision)],
            [sg.Text('Recall: '+recall)],
            [sg.Text('Acuracy: '+accuracy)],
            [sg.Text('F1: '+f1)]]
    
    window2 = sg.Window("Hasil Klasifikasi Sampah", layout2, resizable=True, modal=True)
    window2.read()

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    elif event == "Submit":
        csv_address = values["-FILE_PATH-"]
        print(predict(csv_address))   
window.close()