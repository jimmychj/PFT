from Initialize import init
from extractdata import extract_data
import logging
# from dataproc import process_data
# from RFpredict import predict


def mainexec():
    # sample = input("Enter sample directory: ")
    # sample = 'C:/Users/hc258/PFT Project/sample/'
    sample='sample/'
    hash_data = init(sample)
    # data = input("Enter data directory: ")
    data = 'New FVL/'
    save_dir = 'new DF/'
    extract_data(data, hash_data, save_dir)


if __name__ == '__main__':
    mainexec()

# /Users/jimmy/OneDrive/Documents/Research/CH/PFT/Software/sample/
# /Users/jimmy/OneDrive/Documents/Research/CH/PFT/Software/data/
# AF1061