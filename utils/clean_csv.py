import pandas as pd
import os
import logging
from utils.config import DATA_PATH, DATA_FILE, PARAMS, ORIGINAL_CSV

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def data_df_prep(csv, DATA_FILE):
    data = pd.read_csv(csv)
    logging.info("data loaded") 
    
    """This function is sorting the dataframe columns by date, for easier data processing pipeline. 
    returns sorted dataframe with datetime index. 
    """
    # rename columns
    data.rename(str.lower, axis="columns", inplace=True)
    data.rename(
        columns={
            "use chip": "use_chip",
            "merchant name": "merchant_name",
            "merchant city": "merchant_city",
            "merchant state": "merchant_state",
            "errors?": "errors",
            "is fraud?": "is_fraud",
        },
        inplace=True,
    )

    # convert amount to float (remove $ sign)
    data["amount"] = data["amount"].str[1:].astype("float64")

    # set time series index for sorting
    data[["hour", "minute"]] = data["time"].str.split(":", expand=True)
    data["date"] = pd.to_datetime(
        data[["year", "month", "day", "hour", "minute"]]
    )
    data.set_index("date", inplace=True)
    data.sort_index(inplace=True)
    
    # drop unnecessary column time
    data.drop(columns=["time"], inplace=True)
    try:
        data.to_csv(os.path.join(DATA_FILE,"clean_cc_data.csv"), index_label=False, mode="x")
    except FileExistsError:
        logging.info(f'File exists in folder {DATA_PATH}. Delete it before saving again. \n dataframe created.')
        return data
    
    return data


    

        

if __name__ == "__main__":
    params = PARAMS
    
    data = data_df_prep(os.path.join(DATA_PATH, ORIGINAL_CSV), DATA_PATH) 
    loaded = pd.read_csv(f'data/{DATA_FILE}')
    print(loaded.head(5))
    
    