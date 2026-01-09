import glob
import os

# DEFINE THE TRANSLATION
# Current (Bad) -> Target (COCO Standard)
# 0 (Car)           -> 2 (Car)
# 1 (Pedestrian)    -> 0 (Person)
# 2 (Traffic Light) -> 9 (Traffic Light)
# 3 (Traffic Sign)  -> 11 (Stop Sign - closest match)

id_map = {
    0: 2,
    1: 0,
    2: 9,
    3: 11
}

# POINT TO YOUR DATASET LABELS (Update the path!)
# Example: 'bdd_dataset/valid/labels/*.txt'
label_files = glob.glob("C:\\Users\\rohai\\Desktop\\Final Year Project\\research\\dataset\\valid\\labels\\*.txt") 

print(f"Found {len(label_files)} files. Fixing IDs...")

for file_path in label_files:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        
        try:
            current_id = int(parts[0])
            
            # If we have a translation for this ID, swap it
            if current_id in id_map:
                parts[0] = str(id_map[current_id])
                new_lines.append(" ".join(parts) + "\n")
                
        except ValueError:
            pass # Skip bad lines

    # Overwrite file
    with open(file_path, 'w') as f:
        f.writelines(new_lines)

print("✅ IDs fixed! Cars are now 2, People are now 0.")