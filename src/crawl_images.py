#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LaViC Code: Image Crawling
--------------
A helper script for downloading product images from URLs found
in JSON or JSONL metadata files. Primarily used to populate
train_images/ and valid_images/ directories with product images.
"""

import os
import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from io import BytesIO
from PIL import Image
from tqdm import tqdm


def load_json_data(file_name):
    """
    Load and return JSON data as a Python dictionary.

    Args:
        file_name (str): Path to the JSON file.

    Returns:
        dict: Deserialized data from the JSON file.
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl_data(file_name):
    """
    Load and return a list of Python dict objects from a JSONL file.

    Each line in the file should contain a valid JSON object.

    Args:
        file_name (str): Path to the JSONL file.

    Returns:
        list of dict: Deserialized data for each line in the file.
    """
    data = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_image(url, path):
    """
    Download an image from a given URL and save it to the specified path.
    Optimized for speed with session reuse and faster image handling.

    Args:
        url (str): URL of the image to download.
        path (str): Local file path where the image should be saved.

    Returns:
        bool: True if the image was saved successfully, False otherwise.
    """
    try:
        # Use session for connection reuse
        if not hasattr(save_image, 'session'):
            save_image.session = requests.Session()
            # Configure session for better performance
            save_image.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
        
        response = save_image.session.get(url, timeout=10, stream=True)
        response.raise_for_status()
        
        # Save directly without PIL processing for speed (just copy bytes)
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        return False


def download_single_image(args):
    """
    Download a single image. Used for concurrent downloading.
    
    Args:
        args (tuple): (url, file_path, item_id)
    
    Returns:
        tuple: (success, file_path, item_id)
    """
    url, file_path, item_id = args
    success = save_image(url, file_path)
    return success, file_path, item_id


def download_images_json(item_data, folder_path, max_workers=16):
    """
    Download all images from a JSON dictionary of items using concurrent downloads.

    The dictionary (item_data) maps item IDs to metadata. If 'images'
    is present, each 'large' field is downloaded.

    Args:
        item_data (dict): JSON dictionary containing item info, including 'images'.
        folder_path (str): Destination folder to save images.
        max_workers (int): Maximum number of concurrent download threads.
    """
    os.makedirs(folder_path, exist_ok=True)

    count_exist = 0
    count_save = 0
    count_fail = 0
    lock = Lock()

    # Prepare download tasks
    download_tasks = []
    for item_id, details in item_data.items():
        if 'images' in details:
            for index, img_obj in enumerate(details['images']):
                image_url = img_obj.get('large', '')
                if not image_url:
                    continue
                
                file_path = os.path.join(folder_path, f"{item_id}_{index}.jpg")
                
                if os.path.exists(file_path):
                    with lock:
                        count_exist += 1
                else:
                    download_tasks.append((image_url, file_path, f"{item_id}_{index}"))

    total_to_download = len(download_tasks)
    total_images = total_to_download + count_exist
    
    print(f"[JSON] Found {total_images} images to process in {len(item_data)} items")
    print(f"[JSON] {count_exist} images already exist, {total_to_download} images to download")
    print(f"[JSON] Using {max_workers} concurrent workers")
    
    if total_to_download == 0:
        print(f"[JSON] All images already exist!")
        return
    
    # Create progress bar
    pbar = tqdm(total=total_to_download, desc="Downloading JSON images", unit="img")
    
    start_time = time.time()
    
    # Download with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_task = {executor.submit(download_single_image, task): task for task in download_tasks}
        
        # Process completed downloads with batch updates
        completed_count = 0
        for future in as_completed(future_to_task):
            success, file_path, item_id = future.result()
            
            with lock:
                if success:
                    count_save += 1
                else:
                    count_fail += 1
                completed_count += 1
                
                # Update progress every 10 downloads to reduce overhead
                if completed_count % 10 == 0 or completed_count == total_to_download:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    speed = (count_save + count_fail) / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({
                        'Exist': count_exist, 
                        'Saved': count_save, 
                        'Failed': count_fail,
                        'Speed': f'{speed:.1f}/s'
                    })
            
            pbar.update(1)
    
    pbar.close()
    
    elapsed_time = time.time() - start_time
    avg_speed = (count_save + count_fail) / elapsed_time if elapsed_time > 0 else 0
    print(f"[JSON] Completed in {elapsed_time:.2f}s - Images already exist: {count_exist}, Newly saved: {count_save}, Failed: {count_fail}")
    print(f"[JSON] Average download speed: {avg_speed:.2f} images/second")


def download_images_jsonl(item_data, folder_path, max_workers=16):
    """
    Download images from a JSONL list of items using concurrent downloads.

    Each element is a dict that may include 'image_name' and 'image'.

    Args:
        item_data (list of dict): List of items from a JSONL file.
        folder_path (str): Destination folder to save images.
        max_workers (int): Maximum number of concurrent download threads.
    """
    os.makedirs(folder_path, exist_ok=True)

    count_exist = 0
    count_save = 0
    count_fail = 0
    lock = Lock()

    # Prepare download tasks
    download_tasks = []
    for entry in item_data:
        image_url = entry.get('image', '')
        image_name = entry.get('image_name', '')

        if not image_url or not image_name:
            continue

        file_path = os.path.join(folder_path, image_name)

        if os.path.exists(file_path):
            with lock:
                count_exist += 1
        else:
            download_tasks.append((image_url, file_path, image_name))

    total_to_download = len(download_tasks)
    total_images = total_to_download + count_exist
    
    print(f"[JSONL] Found {total_images} images to process from {len(item_data)} entries")
    print(f"[JSONL] {count_exist} images already exist, {total_to_download} images to download")
    print(f"[JSONL] Using {max_workers} concurrent workers")
    
    if total_to_download == 0:
        print(f"[JSONL] All images already exist!")
        return
    
    # Create progress bar
    pbar = tqdm(total=total_to_download, desc="Downloading JSONL images", unit="img")
    
    start_time = time.time()

    # Download with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_task = {executor.submit(download_single_image, task): task for task in download_tasks}
        
        # Process completed downloads with batch updates
        completed_count = 0
        for future in as_completed(future_to_task):
            success, file_path, image_name = future.result()
            
            with lock:
                if success:
                    count_save += 1
                else:
                    count_fail += 1
                completed_count += 1
                
                # Update progress every 10 downloads to reduce overhead
                if completed_count % 10 == 0 or completed_count == total_to_download:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    speed = (count_save + count_fail) / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({
                        'Exist': count_exist, 
                        'Saved': count_save, 
                        'Failed': count_fail,
                        'Speed': f'{speed:.1f}/s'
                    })
            
            pbar.update(1)
    
    pbar.close()
    
    elapsed_time = time.time() - start_time
    avg_speed = (count_save + count_fail) / elapsed_time if elapsed_time > 0 else 0
    print(f"[JSONL] Completed in {elapsed_time:.2f}s - Images already exist: {count_exist}, Newly saved: {count_save}, Failed: {count_fail}")
    print(f"[JSONL] Average download speed: {avg_speed:.2f} images/second")


def main():
    """
    Main function to orchestrate image downloads for train (JSON) and valid (JSONL) sets.

    - Loads item2meta_train.json and downloads images to ./data/train_images
    - Loads item2meta_valid.jsonl and downloads images to ./data/valid_images
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="LaViC Image Crawler with concurrent downloads")
    parser.add_argument("--workers", type=int, default=64, 
                       help="Number of concurrent download threads (default: 20)")
    parser.add_argument("--train-only", action="store_true", 
                       help="Only download training images")
    parser.add_argument("--valid-only", action="store_true", 
                       help="Only download validation images")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🖼️  LaViC Image Crawler Starting...")
    print(f"🚀 Using {args.workers} concurrent workers")
    print("=" * 60)
    
    total_start_time = time.time()
    
    # 1) Download images for train data (JSON)
    if not args.valid_only:
        train_json_path = './data/item2meta_train.json'
        train_images_dir = './data/train_images'
        if os.path.exists(train_json_path):
            print(f"\n📁 Processing training data: {train_json_path}")
            print(f"📂 Output directory: {train_images_dir}")
            item_data = load_json_data(train_json_path)
            download_images_json(item_data, train_images_dir, max_workers=args.workers)
        else:
            print(f"❌ No file found at {train_json_path}, skipping train images.")

    if not args.train_only:
        print("\n" + "-" * 60)

        # 2) Download images for valid data (JSONL)
        valid_jsonl_path = './data/item2meta_valid.jsonl'
        valid_images_dir = './data/valid_images'
        if os.path.exists(valid_jsonl_path):
            print(f"\n📁 Processing validation data: {valid_jsonl_path}")
            print(f"📂 Output directory: {valid_images_dir}")
            item_data_list = load_jsonl_data(valid_jsonl_path)
            download_images_jsonl(item_data_list, valid_images_dir, max_workers=args.workers)
        else:
            print(f"❌ No file found at {valid_jsonl_path}, skipping valid images.")
    
    total_elapsed_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print(f"✅ All downloads completed in {total_elapsed_time:.2f} seconds!")
    print(f"⚡ Average overall speed: {81071/total_elapsed_time:.2f} images/second" if total_elapsed_time > 0 else "")
    print("=" * 60)


if __name__ == '__main__':
    main()
