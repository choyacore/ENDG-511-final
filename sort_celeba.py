import os, cv2, csv
#https://www.kaggle.com/discussions/getting-started/476509
# Sort CelebA images into 4 classes based on hair color and resize to 64x64.
# This is to create a balanced dataset for training.
# We ended up only using 4 classes (black, blonde, brown, gray) .didn't hve enough time
# We also limited to 3000 images per class to keep the dataset balanced. 
# The resized images are saved in separate folders for each class under './dataset'.
image_dir  = './dataset/img_align_celeba/img_align_celeba'
attributes = './dataset/list_attr_celeba.csv'
out_base = './dataset'
max_per_class = 3000

with open(attributes) as f:
    reader = csv.DictReader(f)
    counts = {'black': 0, 'blonde': 0, 'brown': 0, 'gray': 0} # tracks how many images per class
    for row in reader: # iterate over each image's attributes
        black  = row['Black_Hair'].strip() == '1'
        blonde = row['Blond_Hair'].strip() == '1'
        brown  = row['Brown_Hair'].strip() == '1'
        gray   = row['Gray_Hair'].strip()  == '1'
    # Assign class based on hair color adn makes sure it's within max per class limit
        if black and counts['black'] < max_per_class:
            cls = 'black'
        elif blonde and counts['blonde'] < max_per_class:
            cls = 'blonde'
        elif brown and counts['brown'] < max_per_class:
            cls = 'brown'
        elif gray and counts['gray'] < max_per_class:
            cls = 'gray'
        else:
            continue
            # Skip if no class assigned 
        fname = row['image_id'].strip()
        source   = os.path.join(image_dir, fname)
        if not os.path.exists(source):
            continue
            # Skip if no image 
        dst_dir = os.path.join(out_base, cls)
        os.makedirs(dst_dir, exist_ok=True)
        img = cv2.imread(source)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))
        cv2.imwrite(os.path.join(dst_dir, f'celeba_{fname}'), img)
        counts[cls] += 1

        if sum(counts.values()) % 1000 == 0:
            print(f'Progress: {counts}')

print(f'Done! {counts}')