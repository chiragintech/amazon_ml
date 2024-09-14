import pandas as pd
import matplotlib.pyplot as plt
import easyocr
import re
from PIL import Image

data = pd.read_csv('test.csv')
reader = easyocr.Reader(['en'], gpu=True)

ocr_results = []

def infer_entity(entity_name, number):
    number = float(number)
    if entity_name == 'height':
        if 100 <= number <= 300:
            return True
    elif entity_name == 'weight':
        if 20 <= number <= 200:
            return True
    elif entity_name == 'voltage':
        if 110 <= number <= 240:
            return True
    elif entity_name == 'depth' or entity_name == 'width':
        if 10 <= number <= 300:
            return True
    return False

for idx, row in data.iterrows():
    image_path = row['image_link']
    group_id = row['group_id']
    entity_name = row['entity_name'].lower()
    
    try:
        img = Image.open(image_path)
        
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'{entity_name.capitalize()} - Group: {group_id}')
        plt.show()

        results = reader.readtext(image_path)
        number_pattern = r'\d+(\.\d+)?'

        found = False

        for res in results:
            bbox, text, conf = res
            numbers_in_text = re.findall(number_pattern, text)

            for number in numbers_in_text:
                if infer_entity(entity_name, number):
                    ocr_results.append({
                        'index': idx,
                        'group_id': group_id,
                        'entity_name': entity_name,
                        'text': text,
                        'number': number,
                        'confidence': conf,
                        'bbox': bbox
                    })
                    found = True
                    break

        if not found:
            print(f"Could not find a valid '{entity_name}' value in image {image_path}")

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

ocr_df = pd.DataFrame(ocr_results)
ocr_df.to_csv('ocr_results.csv', index=False)
