import pandas as pd
import matplotlib.pyplot as plt
import easyocr
from PIL import Image

data = pd.read_csv('test.csv')

reader = easyocr.Reader(['en'], gpu=True)

ocr_results = []

for idx, row in data.iterrows():
    image_path = row['image_link']
    group_id = row['group_id']
    entity_name = row['entity_name']
    
    try:
        img = Image.open(image_path)

        plt.imshow(img)
        plt.axis('off')
        plt.title(f'{entity_name} - Group: {group_id}')
        plt.show()

        results = reader.readtext(image_path)
        
        for res in results:
            bbox, text, conf = res
            ocr_results.append({
                'index': idx,
                'group_id': group_id,
                'entity_name': entity_name,
                'bbox': bbox,
                'text': text,
                'confidence': conf
            })
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

ocr_df = pd.DataFrame(ocr_results)

ocr_df.to_csv('ocr_results.csv', index=False)

