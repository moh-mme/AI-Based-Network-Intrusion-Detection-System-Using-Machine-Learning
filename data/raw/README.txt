Put the CIC-IDS2017 CSV files in this folder.

1. Request the dataset at https://www.unb.ca/cic/datasets/ids-2017.html
2. Download MachineLearningCSV.zip
3. Extract it here. Sub-folders are fine; every *.csv below this folder is read.

Expected files (8 in total):
  Monday-WorkingHours.pcap_ISCX.csv
  Tuesday-WorkingHours.pcap_ISCX.csv
  Wednesday-workingHours.pcap_ISCX.csv
  Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
  Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
  Friday-WorkingHours-Morning.pcap_ISCX.csv
  Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
  Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

Then run:  python -m nids.prepare
