import kagglehub
from kagglehub import KaggleDatasetAdapter
from pathlib import Path
import pandas as pd

# Set the path to the file you'd like to load
file_path = "cardio_train.csv"

# Load the latest version
sulianova_data = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "sulianova/cardiovascular-disease-dataset",
    file_path,
    pandas_kwargs={"sep": ";", "names": ["id", "age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]}
)

def load_echonet_with_kagglehub():
    """Load EchoNet-Dynamic dataset using kagglehub"""
    try:
        # Install kagglehub if not available
        try:
            import kagglehub
            from kagglehub import KaggleDatasetAdapter
        except ImportError:
            print("Installing kagglehub...")
            import subprocess
            subprocess.run(['pip', 'install', 'kagglehub[pandas-datasets]'], check=True)
            import kagglehub
            from kagglehub import KaggleDatasetAdapter

        print("Loading EchoNet-Dynamic dataset from Kaggle...")

        # Download the dataset - this will download all files including Videos folder
        dataset_path = kagglehub.dataset_download("mahnurrahman/echonet-dynamic")
        print(f"Dataset downloaded to: {dataset_path}")

        # The dataset should now be available at the downloaded path
        dataset_path = Path(dataset_path)

        # Check structure
        # print("Dataset structure:")
        # for item in dataset_path.rglob("*"):
        #     if item.is_file():
        #         print(f"  File: {item.relative_to(dataset_path)}")
        #     elif item.is_dir():
        #         print(f"  Dir: {item.relative_to(dataset_path)}/")

        # Load FileList.csv
        filelist_path = None
        volume_tracings_path = None
        videos_path = None

        # Find the CSV files and Videos folder
        for item in dataset_path.rglob("*"):
            if item.name == "FileList.csv":
                filelist_path = item
            elif item.name == "VolumeTracings.csv":
                volume_tracings_path = item
            elif item.name == "Videos" and item.is_dir():
                videos_path = item

        if filelist_path is None:
            print("FileList.csv not found in downloaded dataset")
            return None, None, None

        # Load metadata
        filelist_df = pd.read_csv(filelist_path)
        print(f"FileList.csv loaded: {filelist_df.shape}")
        print(f"Columns: {filelist_df.columns.tolist()}")

        volume_df = None
        if volume_tracings_path:
            volume_df = pd.read_csv(volume_tracings_path)
            print(f"VolumeTracings.csv loaded: {volume_df.shape}")

        return filelist_df, volume_df, videos_path

    except Exception as e:
        print(f"Error loading EchoNet data: {e}")
        print("Falling back to alternative loading method...")

        # Alternative: try direct pandas loading
        try:
            df = kagglehub.load_dataset( # type: ignore
                KaggleDatasetAdapter.PANDAS, # type: ignore
                "mahnurrahman/echonet-dynamic",
                "",
            )
            print("Loaded EchoNet data as DataFrame:")
            print("First 5 records:", df.head())
            return df, None, None
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            return None, None, None