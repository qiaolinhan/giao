from pdf2image import conver_form_path

page = conver_form_path("giao/unet.pdf")
page.save("unetstructure.jpg", "JPEG")
