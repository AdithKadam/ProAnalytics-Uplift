"""
Download Dunnhumby Complete Journey dataset from Kaggle
"""
import kagglehub
import shutil
import os
import papermill as pm

print("Downloading Dunnhumby Complete Journey dataset...")
path = kagglehub.dataset_download("frtgnn/dunnhumby-the-complete-journey")
print(f"Downloaded to: {path}")

# Move files to data/ folder
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

for file in os.listdir(path):
    if file.endswith('.csv'):
        src = os.path.join(path, file)
        dst = os.path.join(data_dir, file)
        shutil.copy2(src, dst)
        print(f"Copied {file} to {data_dir}/")

print("\n✅ Dataset ready in data/ folder!")

# Run eda_dunnhumby.ipynb to perform EDA and save results to eda_dunnhumby_output_notebook.ipynb
print("\nRunning EDA notebook (notebooks/eda_dunnhumby.ipynb)")
pm.execute_notebook(
   'notebooks/eda_dunnhumby.ipynb',
   'notebooks/eda_dunnhumby_output.ipynb',
)
print("\nOutput saved to notebooks/eda_dunnhumby_output.ipynb")