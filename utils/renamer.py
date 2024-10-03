import os


target_dir = '/scr/user/mahdinur/CVPR/2022-CVPR-AirNet/test/derain/target'


for filename in os.listdir(target_dir):
    
    
    old_file_path = os.path.join(target_dir, filename)

    
    if os.path.isfile(old_file_path):
        
        base_name, ext = os.path.splitext(filename)

        
        number = base_name.split('-')[-1]

        
        new_filename = f"rain-{number}{ext}"
        new_file_path = os.path.join(target_dir, new_filename)
        print(f'{filename} {new_filename}')
        os.rename(old_file_path, new_file_path)

print("Renaming completed.")
