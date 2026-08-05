import pandas as pd
import numpy as np
from PIL import Image
from wordcloud import WordCloud
import matplotlib.font_manager as fm

# ============================================================
# Config
# ============================================================
# CSV column names (must match the actual CSV files, unchanged from source data)
COL_TEXT     = '英文'      # column with the (translated) English words
COL_GROUP    = '组别'      # column with the group label
COL_WEIGHT   = '权重'      # column with word weight/frequency
GROUP_DEP    = '抑郁'
GROUP_NODEP  = '非抑郁'

# Colors
COLOR_DEP          = '#AFA0D0'   # depression group: normal word
COLOR_DEP_HL       = '#583A91'   # depression group: highlighted word
COLOR_NODEP        = '#B2CEBF'   # non-depression group: normal word
COLOR_NODEP_HL     = '#457F63'   # non-depression group: highlighted word

# Word cloud parameters
MAX_WORDS    = 60
MIN_FONT     = 16
MAX_FONT     = 200
PREFER_HORIZ = 1.0
REL_SCALING  = 0.5
MASK_SIZE    = 1200
FONT         = 'Arial'
FONT_PATH    = fm.findfont(fm.FontProperties(family=FONT))
GAP          = 40

# PNG output parameters
SCALE        = 2     # pixel upscaling only; layout is computed at MASK_SIZE and is unaffected
DPI          = 300   # metadata written into the PNG header

# One entry per figure to generate
TOPICS = [
    {
        'name':            'greenspace',
        'csv_path':        '../../Data/LooMen/Data_4_wordcloud_greenspace.csv',
        'output_png':      '../../Results/Figure2/Figure2d_greenspace.png',
        'highlight_dep':   ['Desire', 'Hideaway', 'Inner', 'Aspiring',
                             'Warmth', 'Voice', 'Activist', 'Attachment'],
        'highlight_nodep': ['Daily', 'Casually', 'Foliage', 'Breathe', 'Eyeful',
                             'Earth', 'Nature', 'Living', 'Fragment'],
        'seed_dep':        2,
        'seed_nodep':      5,
    },
    {
        'name':            'temple',
        'csv_path':        '../../Data/LooMen/Data_4_wordcloud_temple.csv',
        'output_png':      '../../Results/Figure2/Figure2d_temple.png',
        'highlight_dep':   ['Release', 'Solace', 'Hardship', 'Wish', 'Cultivation', 'Happiness',
                             'Accept', 'Hope', 'Spirit', 'Gratitude'],
        'highlight_nodep': ['Located', 'Guide', 'Built', 'Exquisite', 'Daily', 'Scale',
                             'Famous', 'Super-topic', 'Photography', 'Travel', 'Incense'],
        'seed_dep':        5,
        'seed_nodep':      3,
    },
]

# ============================================================

def create_circle_mask(size):
    mask = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    y, x = np.ogrid[:size, :size]
    mask[(x - c) ** 2 + (y - c) ** 2 <= (c - 10) ** 2] = 0
    return mask

def make_color_func(highlight_list, color_normal, color_highlight):
    hl_set = {w.lower() for w in highlight_list}
    def color_func(word, **kwargs):
        return color_highlight if word.lower() in hl_set else color_normal
    return color_func

def make_wordcloud(freq_dict, color_func, mask, seed):
    if not freq_dict:
        freq_dict = {'No Significant Words': 1.0}
    return WordCloud(
        width=MASK_SIZE, height=MASK_SIZE,
        background_color='white',
        mask=mask,
        max_words=MAX_WORDS,
        max_font_size=MAX_FONT,
        min_font_size=MIN_FONT,
        font_path=FONT_PATH,
        relative_scaling=REL_SCALING,
        prefer_horizontal=PREFER_HORIZ,
        random_state=seed,
        color_func=color_func,
        scale=SCALE,
        mode='RGB',
    ).generate_from_frequencies(freq_dict)

def merge_two_images(img_left, img_right, gap=GAP):
    gap_px = int(round(gap * SCALE))
    total_w = img_left.width + gap_px + img_right.width
    total_h = max(img_left.height, img_right.height)
    canvas = Image.new('RGB', (total_w, total_h), 'white')
    canvas.paste(img_left, (0, 0))
    canvas.paste(img_right, (img_left.width + gap_px, 0))
    return canvas

def generate_figure(topic, mask):
    df = pd.read_csv(topic['csv_path'])
    if COL_TEXT not in df.columns or df[COL_TEXT].isna().all():
        print(f"[{topic['name']}] Error: please run the translation script first")
        return

    df = df[df[COL_TEXT].notna() & (df[COL_TEXT].str.strip() != '')].copy()
    df[COL_TEXT] = df[COL_TEXT].str.capitalize()

    freq_dep   = dict(zip(df[df[COL_GROUP] == GROUP_DEP][COL_TEXT],   df[df[COL_GROUP] == GROUP_DEP][COL_WEIGHT]))
    freq_nodep = dict(zip(df[df[COL_GROUP] == GROUP_NODEP][COL_TEXT], df[df[COL_GROUP] == GROUP_NODEP][COL_WEIGHT]))

    wc_dep   = make_wordcloud(freq_dep,   make_color_func(topic['highlight_dep'],   COLOR_DEP,   COLOR_DEP_HL),   mask, topic['seed_dep'])
    wc_nodep = make_wordcloud(freq_nodep, make_color_func(topic['highlight_nodep'], COLOR_NODEP, COLOR_NODEP_HL), mask, topic['seed_nodep'])

    merged = merge_two_images(wc_dep.to_image(), wc_nodep.to_image())
    merged.save(topic['output_png'], format='PNG', dpi=(DPI, DPI))
    print(f"[{topic['name']}] PNG saved: {topic['output_png']}")

def main():
    mask = create_circle_mask(MASK_SIZE)
    for topic in TOPICS:
        generate_figure(topic, mask)

if __name__ == '__main__':
    main()