import json

image_id = []

# Load JSON data from file
with open("output.json", "r") as f:
    data = json.load(f)

#Keeping track of unique IDs
def keyID():   
    number_map = {}
    new_number = 0

    for item in image_id:
        if item in number_map:
            item = number_map[item]
        else:
            number_map[item] = new_number
            item = new_number
            new_number += 1
    return number_map

# Loop through each element in the "features" list
#Make changes to JSON file
for feature in data["annotations"]:

    feature["segmentation"] = []

    area = feature["area"]
    feature["area"] = round(area, 2)
    
    category_id = feature["category_id"]
    feature["category_id"] = 0

    image_id.append(feature["image_id"])
    idKeys = keyID()
    feature["image_id"] = idKeys[feature["image_id"]]

    bbox = feature["bbox"]
    for i in range(len(bbox)):
        bbox[i] = round(bbox[i], 2)

for feature in data["categories"]:
    feature["id"] = 0

for feature in data["images"]:
    feature["id"] = idKeys[feature["id"]]

# Write the updated data back to the file
with open("custom_train.json", "w") as f:
    json.dump(data, f, indent=2)