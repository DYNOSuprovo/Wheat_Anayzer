import zipfile
zip_file_path = r'D:\SIH\SIHV2\wheat_analysis_app\archive (2).zip'
extract_path = r'D:\SIH\SIHV2\wheat_analysis_app\data\classification\Images'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)