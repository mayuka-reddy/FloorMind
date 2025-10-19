#!/usr/bin/env python3
"""
CubiCasa5K Dataset Downloader
Simplified script to download and setup CubiCasa5K dataset
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

def create_data_folder():
    """Create data folder structure"""
    print("📁 Creating data folder structure...")
    
    base_dir = Path(__file__).parent
    cubicasa_dir = base_dir / "cubicasa5k"
    
    # Create directories
    cubicasa_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created directory: {cubicasa_dir}")
    return cubicasa_dir

def download_file(url, output_path, description):
    """Download file with progress bar"""
    print(f"⬇️  Downloading {description}...")
    print(f"📍 URL: {url}")
    print(f"💾 Output: {output_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=f"Downloading {description}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ Downloaded {description} successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {description}: {e}")
        return False

def extract_zip(zip_path, extract_dir, description):
    """Extract zip file"""
    print(f"📂 Extracting {description}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files for progress bar
            file_list = zip_ref.namelist()
            
            with tqdm(total=len(file_list), desc=f"Extracting {description}") as pbar:
                for file in file_list:
                    zip_ref.extract(file, extract_dir)
                    pbar.update(1)
        
        print(f"✅ Extracted {description} successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error extracting {description}: {e}")
        return False

def download_cubicasa5k():
    """Download CubiCasa5K dataset"""
    
    print("🏗️  FloorMind CubiCasa5K Dataset Downloader")
    print("=" * 60)
    
    # Step 1: Create data folder
    cubicasa_dir = create_data_folder()
    
    # Dataset URLs
    urls = {
        'images': 'https://zenodo.org/record/2613548/files/cubicasa5k.zip',
        'annotations': 'https://zenodo.org/record/2613548/files/cubicasa5k_annotations.zip'
    }
    
    # Step 2: Download both archives
    print("\n📦 Downloading CubiCasa5K dataset files...")
    
    downloads = []
    for file_type, url in urls.items():
        output_file = cubicasa_dir / f"cubicasa5k_{file_type}.zip"
        
        # Check if already downloaded
        if output_file.exists():
            print(f"⚠️  {file_type} already exists: {output_file}")
            print("💡 Delete the file to redownload")
            downloads.append((output_file, file_type))
            continue
        
        success = download_file(url, output_file, f"CubiCasa5K {file_type}")
        if success:
            downloads.append((output_file, file_type))
        else:
            print(f"❌ Failed to download {file_type}")
            return False
    
    # Step 3: Extract both archives
    print("\n📂 Extracting downloaded files...")
    
    for zip_file, file_type in downloads:
        if zip_file.exists():
            success = extract_zip(zip_file, cubicasa_dir, f"CubiCasa5K {file_type}")
            if not success:
                return False
            
            # Remove zip file after extraction to save space
            print(f"🗑️  Removing {zip_file.name} to save space...")
            zip_file.unlink()
    
    # Step 4: Create dataset info
    info = {
        'name': 'CubiCasa5K',
        'description': 'Large-scale floor plan dataset with 5000+ annotated floor plans',
        'download_date': datetime.now().isoformat(),
        'source': 'https://zenodo.org/record/2613548',
        'citation': 'Kalervo, A., Kämppi, M., Lehtiniemi, T., & Rantanen, T. (2019). CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis.',
        'license': 'CC BY 4.0',
        'files': {
            'images': 'Floor plan images in PNG format',
            'annotations': 'JSON annotations with room segmentation and metadata'
        }
    }
    
    info_file = cubicasa_dir / 'dataset_info.json'
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    # Step 5: Analyze dataset structure
    print("\n🔍 Analyzing dataset structure...")
    analyze_dataset_structure(cubicasa_dir)
    
    print("\n" + "=" * 60)
    print("🎉 CubiCasa5K Dataset Setup Complete!")
    print("=" * 60)
    print(f"📁 Dataset location: {cubicasa_dir}")
    print(f"📄 Dataset info: {info_file}")
    
    print("\n💡 Next steps:")
    print("1. Process the dataset:")
    print("   python dataset_manager.py process --dataset cubicasa5k")
    print("2. Create train/test splits:")
    print("   python dataset_manager.py split")
    print("3. Generate dataset report:")
    print("   python dataset_manager.py report")
    
    return True

def analyze_dataset_structure(cubicasa_dir):
    """Analyze the downloaded dataset structure"""
    
    print("📊 Dataset Structure Analysis:")
    
    # Count directories and files
    total_dirs = 0
    total_files = 0
    image_files = 0
    json_files = 0
    
    for root, dirs, files in os.walk(cubicasa_dir):
        total_dirs += len(dirs)
        total_files += len(files)
        
        for file in files:
            if file.endswith('.png'):
                image_files += 1
            elif file.endswith('.json'):
                json_files += 1
    
    print(f"   📁 Total directories: {total_dirs:,}")
    print(f"   📄 Total files: {total_files:,}")
    print(f"   🖼️  Image files (.png): {image_files:,}")
    print(f"   📋 JSON files (.json): {json_files:,}")
    
    # Find main directories
    main_dirs = [d for d in cubicasa_dir.iterdir() if d.is_dir()]
    print(f"   🗂️  Main directories: {[d.name for d in main_dirs]}")
    
    # Estimate dataset size
    total_size = sum(f.stat().st_size for f in cubicasa_dir.rglob('*') if f.is_file())
    size_gb = total_size / (1024**3)
    print(f"   💾 Total size: {size_gb:.2f} GB")

def main():
    """Main function"""
    
    try:
        success = download_cubicasa5k()
        if success:
            print("\n✅ All operations completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Some operations failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()