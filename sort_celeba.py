import os, cv2, csv
#https://www.kaggle.com/discussions/getting-started/476509


# We ended up only using 4 classes (black, blonde, brown, gray) .didn't hve enough time
# We used 3000 images per class to keep the dataset balanced

image_dir  = './dataset/img_align_celeba/img_align_celeba'
attributes = './dataset/list_attr_celeba.csv'
out_base = './dataset'
max_per_class = 3000

with open(attributes) as f:
    reader = csv.DictReader(f)
    count_img = {'black': 0, 'blonde': 0, 'brown': 0, 'gray': 0} # tracks how many images per class
    for row in reader: # iterate over each image's attributes
        black  = row['Black_Hair'].strip() == '1'
        blonde = row['Blond_Hair'].strip() == '1'
        brown  = row['Brown_Hair'].strip() == '1'
        gray   = row['Gray_Hair'].strip()  == '1'
    # Assign class based on hair color adn makes sure it's within max per class limit
        if black and count_img['black'] < max_per_class:
            hair_class = 'black'
        elif blonde and count_img['blonde'] < max_per_class:
            hair_class = 'blonde'
        elif brown and count_img['brown'] < max_per_class:
            hair_class = 'brown'
        elif gray and count_img['gray'] < max_per_class:
            hair_class = 'gray'
        else:
            continue
            # Skip if no class assigned 
        file_name = row['image_id'].strip()
        source   = os.path.join(image_dir, file_name)
        if not os.path.exists(source):
            continue
            # Skip if no image 
        dst_dir = os.path.join(out_base, hair_class)
        os.makedirs(dst_dir, exist_ok=True)
        img = cv2.imread(source)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))
        cv2.imwrite(os.path.join(dst_dir, f'celeba_{file_name}'), img)
        count_img[hair_class] += 1

        if sum(count_img.values()) % 1000 == 0:
            print(f'Progress: {count_img}')

print(f'Done! {count_img}')