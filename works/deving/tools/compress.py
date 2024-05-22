import os
from PIL import Image
from PIL import ImageFile

def compress(outfile, kb=10, quality=85, k=0.9):

    o_size = os.path.getsize(outfile) // 1024
    if o_size <= kb:
        return outfile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    while o_size > kb:
        im = Image.open(outfile)
        x, y = im.size
        out = im.resize((int(x * k), int(y * k)), Image.LANCZOS)
        try:
            out.save(outfile, quality = quality)
        except Exception as e:
            print(e)
            break

        o_size = os.path.getsize(outfile) // 1024

    return outfile

outfile = "./1.JPG"
compress(outfile)
