import pandas as pd
import os
import shutil

class DataManager:
    def __init__(self, manifest_path):
        self.manifest_path = manifest_path
        if os.path.exists(manifest_path):
            self.df = pd.read_csv(manifest_path)
        else:
            self.df = pd.DataFrame(columns=['file_path', 'file_name', 'label', 'type', 'split'])
            
    def get_manifest(self):
        return self.df
    
    def save_manifest(self):
        self.df.to_csv(self.manifest_path, index=False)
        
    def add_files(self, uploaded_files, label, split='train', save_dir='data/uploads'):
        """
        Streamlit UploadedFile objects handling
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        new_rows = []
        for uploaded_file in uploaded_files:
            # Save file to disk
            file_path = os.path.join(save_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            new_rows.append({
                'file_path': os.path.abspath(file_path),
                'file_name': uploaded_file.name,
                'label': label,
                'type': 'image' if uploaded_file.type.startswith('image') else 'omics',
                'split': split
            })
            
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            self.save_manifest()
            return True
        return False

    def get_stats(self):
        return self.df['label'].value_counts()
