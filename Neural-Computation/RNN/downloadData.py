from huggingface_hub import HfFileSystem
from huggingface_hub import hf_hub_url
import webdataset as wds

fs = HfFileSystem()
files = [fs.resolve_path(path) for path in fs.glob("hf://datasets/aisuko/ucf101-subset/UCF101_subset.tar.gz")]
print(files)

urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]
print(urls)

ds = wds.WebDataset(urls).decode()
for sample in ds:
    print(sample)