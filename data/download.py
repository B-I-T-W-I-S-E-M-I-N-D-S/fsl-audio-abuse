import gdown 

def download_dataset():
    ids = [
        "1DGTttJ7_pHOweto7129gni4nb9p0Iux7",
        "1BHLOCEtofq8KyI80n1H3nWpd2tvpyEyw",
        "1uKv79BKuCJmfEWe2MMWPbofRAVWuLRqw",
        "1GyBOIoIz3mSygMYYpdcA2VeanIZJSLCM",
    ]
    for id in ids:
        gdown.download(id=id, output="./datasets")
    print("Download Done")