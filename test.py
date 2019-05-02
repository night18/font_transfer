import os
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

TEXTS_DIR = "texts"
IMAGES_DIR = "font"
TTF_NAME = "wt024"
TTF_PATH = "wt024.ttf"
# TTF_NAME = "wt071"
# TTF_PATH = "wt071.ttf"
FONT_SIZE = "300"


ttf = TTFont(TTF_PATH)
print (ttf)
# for x in ttf["cmap"].tables:
#     for y in x.cmap.items():
#         char_unicode = chr(y[0])
#         char_utf8 = char_unicode.encode('utf_8')
#         # print(char_utf8)

#         char_name = y[1]
#         f = open(os.path.join(TEXTS_DIR, char_name + '.txt'), 'wb')
#         f.write(char_utf8)
#         f.close()
# ttf.close()

font = ImageFont.truetype(TTF_PATH, 300)
# files = os.listdir(TEXTS_DIR)
files = ['uni6C38.txt', 'uni548C.txt','uni4E5D.txt', 'uni5E74.txt']
idx = 0
for filename in files:
    name, ext = os.path.splitext(filename)
    input_txt = TEXTS_DIR + "/" + filename
    output_png = IMAGES_DIR + "/" + TTF_NAME + '/'+ name + "_" + FONT_SIZE  +".png"

    img = Image.new('RGB', (300, 300), color='white')
    d = ImageDraw.Draw(img)

    with open(os.path.join(TEXTS_DIR, filename)) as f:
        c = f.read(1)
        d.text((0,0), c, fill=(0,0,0), font=font)
        img.save(output_png)

    # idx += 1
    # if idx > 10000:
    #     break

# #     subprocess.call(["convert", "-font", TTF_PATH, "-pointsize", FONT_SIZE, "-background", "rgba(0,0,0,0)", "label:@" + input_txt, output_png])

print("finished")