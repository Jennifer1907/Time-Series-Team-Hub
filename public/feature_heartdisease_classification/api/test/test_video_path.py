"""
Test script to verify EchoNet video path resolution
"""
from utility.load_data import load_echonet_with_kagglehub
from pathlib import Path

print("="*70)
print("TESTING ECHONET VIDEO PATH RESOLUTION")
print("="*70)

# Load EchoNet data
echonet_filelist, echonet_volume, videos_path = load_echonet_with_kagglehub()

if echonet_filelist is not None and videos_path is not None:
    print(f"\n✓ Successfully loaded EchoNet data")
    print(f"  - FileList entries: {len(echonet_filelist)}")
    print(f"  - Videos folder: {videos_path}")
    print(f"  - Videos folder exists: {videos_path.exists()}")

    # Test first 5 videos
    print(f"\n{'='*70}")
    print("TESTING FIRST 5 VIDEO FILES")
    print("="*70)

    for idx in range(min(5, len(echonet_filelist))):
        video_filename = echonet_filelist.iloc[idx]['FileName']

        # Add .avi extension if needed
        if not video_filename.endswith('.avi'):
            video_filename_with_ext = f"{video_filename}.avi"
        else:
            video_filename_with_ext = video_filename

        video_path = videos_path / video_filename_with_ext

        exists = video_path.exists()
        status = "✓" if exists else "✗"

        print(f"\n[{idx}] {status} {video_filename}")
        print(f"    Full name: {video_filename_with_ext}")
        print(f"    Path: {video_path}")
        print(f"    Exists: {exists}")

        # If doesn't exist, try to find it
        if not exists:
            # Check if file exists without extension
            path_no_ext = videos_path / video_filename
            print(f"    Without .avi: {path_no_ext.exists()}")

            # Check what files actually exist
            matching_files = list(videos_path.glob(f"{video_filename}*"))
            if matching_files:
                print(f"    Found similar files: {[f.name for f in matching_files]}")

    # Count total .avi files
    print(f"\n{'='*70}")
    print("VIDEO FILES SUMMARY")
    print("="*70)
    avi_files = list(videos_path.glob("*.avi"))
    print(f"Total .avi files in Videos folder: {len(avi_files)}")

    if avi_files:
        print(f"\nFirst 3 actual files:")
        for f in avi_files[:3]:
            print(f"  - {f.name}")

else:
    print("✗ Failed to load EchoNet data")
