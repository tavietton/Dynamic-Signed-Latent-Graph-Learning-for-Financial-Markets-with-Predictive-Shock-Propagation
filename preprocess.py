import os
import glob
import numpy as np
import pandas as pd


DATA_FOLDER = "csv"

OUTPUT_FILE = "features.npy"
META_FILE = "meta.npz"


MIN_DAYS = 2500

EPS = 1e-8


REFERENCE_TICKER = "^GSPC"


SAMPLE_START_DATE = "2015-10-19"
SAMPLE_END_DATE = "2026-05-15"


APPLY_STALE_FILTER = False
MAX_END_GAP = 5


def make_features(df):
    """
    Construct six features from raw OHLCV columns.

    Input df must be indexed by Date and contain:
    Open, High, Low, Close, Adj Close, Volume.

    Features:
        f0 = log(Open_t / Close_{t-1})      overnight return
        f1 = log(Close_t / Open_t)          intraday return
        f2 = log(High_t / Open_t)           high excursion
        f3 = log(Low_t / Open_t)            low excursion
        f4 = log(Close_t / Close_{t-1})     close-to-close return
        f5 = log(1 + Volume_t)              log-volume
    """

    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    close = close.clip(lower=EPS)
    open_ = open_.clip(lower=EPS)
    high = high.clip(lower=EPS)
    low = low.clip(lower=EPS)
    volume = volume.clip(lower=0)

    feat = pd.DataFrame(index=df.index)

    feat["f0"] = np.log(open_ / close.shift(1).clip(lower=EPS))
    feat["f1"] = np.log(close / open_)
    feat["f2"] = np.log(high / open_)
    feat["f3"] = np.log(low / open_)
    feat["f4"] = np.log(close / close.shift(1).clip(lower=EPS))
    feat["f5"] = np.log1p(volume)

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

    return feat


def clean_raw_dataframe(df):
    """
    Clean one raw OHLCV dataframe.

    Expected columns:
        Date, Adj Close, Close, High, Low, Open, Volume
    """

    required_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")

    df = df[
        ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    ]

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    df = df.drop_duplicates(subset=["Date"], keep="last")


    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        df = df[df[col] > 0]

    df = df[df["Volume"] >= 0]
    df = df[df["High"] >= df["Low"]]


    ret = df["Close"].pct_change()
    df = df[(ret.abs() < 0.8) | (ret.isna())]

    df = df.sort_values("Date")

    return df


def load_reference_calendar(folder):
    """
    Load the reference market calendar from REFERENCE_TICKER.
    Only Date is used as the calendar.
    """

    ref_file = os.path.join(folder, f"{REFERENCE_TICKER}.csv")

    if not os.path.exists(ref_file):
        raise FileNotFoundError(
            f"Reference calendar file not found: {ref_file}. "
            f"Please make sure {REFERENCE_TICKER}.csv exists in {folder}/"
        )

    ref_df = pd.read_csv(ref_file)
    ref_df = clean_raw_dataframe(ref_df)
    ref_df = ref_df.set_index("Date").sort_index()

    reference_dates = pd.DatetimeIndex(sorted(ref_df.index.unique()))

    sample_start = pd.Timestamp(SAMPLE_START_DATE)
    sample_end = pd.Timestamp(SAMPLE_END_DATE)

    reference_window_dates = reference_dates[
        (reference_dates >= sample_start) &
        (reference_dates <= sample_end)
    ]

    if len(reference_window_dates) == 0:
        raise ValueError(
            "No reference-calendar dates found inside the fixed sample window. "
            f"Check SAMPLE_START_DATE={SAMPLE_START_DATE} and SAMPLE_END_DATE={SAMPLE_END_DATE}."
        )

    print("=" * 60)
    print("Reference calendar:")
    print("Ticker:", REFERENCE_TICKER)
    print("Full reference dates:", len(reference_dates))
    print("Reference date range:")
    print("From:", reference_dates[0])
    print("To  :", reference_dates[-1])
    print("Fixed sample window:")
    print("From:", reference_window_dates[0])
    print("To  :", reference_window_dates[-1])
    print("Window trading days:", len(reference_window_dates))
    print("=" * 60)

    return reference_dates, reference_window_dates


def load_and_filter_data(folder):

    files = sorted(glob.glob(os.path.join(folder, "*.csv")))

    reference_dates, reference_window_dates = load_reference_calendar(folder)

    reference_dates_set = set(reference_dates)
    reference_window_dates_set = set(reference_window_dates)

    data_dict = {}
    meta_info = {}


    for f in files:

        name = os.path.basename(f).replace(".csv", "")

        try:
            df = pd.read_csv(f)
            df = clean_raw_dataframe(df)
            df = df.set_index("Date").sort_index()


            df = df[df.index.isin(reference_dates_set)]

            if len(df) < MIN_DAYS:
                print(f"SKIP {name}: too short after {REFERENCE_TICKER} calendar alignment ({len(df)} days)")
                continue


            available_window_dates = set(df.index) & reference_window_dates_set
            missing_count = len(reference_window_dates_set) - len(available_window_dates)

            if missing_count > 0:
                print(
                    f"SKIP {name}: missing {missing_count} "
                    f"{REFERENCE_TICKER}-calendar days in fixed sample window"
                )
                continue

            start = df.index.min()
            end = df.index.max()

            meta_info[name] = (start, end, len(df))
            data_dict[name] = df

        except Exception as e:
            print(f"FAILED {name}: {e}")
            continue

    print("Stocks after fixed-window coverage filter:", len(data_dict))

    if len(data_dict) == 0:
        raise RuntimeError(
            "No stocks remain after filtering. "
            "Consider relaxing SAMPLE_START_DATE/SAMPLE_END_DATE or checking input CSV files."
        )


    if APPLY_STALE_FILTER:
        reference_latest_for_stale = pd.Timestamp(SAMPLE_END_DATE)

        final_dict = {}
        stock_names = []

        for k, df in data_dict.items():
            start, end, length = meta_info[k]

            gap = (reference_latest_for_stale - end).days

            if gap > MAX_END_GAP:
                print(f"SKIP {k}: stale relative to sample end (gap {gap} days)")
                continue

            final_dict[k] = df
            stock_names.append(k)

    else:
        final_dict = data_dict
        stock_names = list(final_dict.keys())

    print("Remaining stocks:", len(final_dict))

    if len(final_dict) == 0:
        raise RuntimeError(
            "No stocks remain after optional stale filtering. "
            "Set APPLY_STALE_FILTER=False or relax MAX_END_GAP."
        )


    keys = list(final_dict.keys())

    common_dates = set(final_dict[keys[0]].index)

    for k in keys[1:]:
        common_dates &= set(final_dict[k].index)

    common_dates = sorted(common_dates)
    common_dates = pd.DatetimeIndex(common_dates)


    missing_from_intersection = reference_window_dates.difference(common_dates)

    if len(missing_from_intersection) > 0:
        raise RuntimeError(
            "Internal consistency error: final intersection does not contain "
            "all fixed-window reference dates, even though coverage filtering passed. "
            f"Missing dates count: {len(missing_from_intersection)}"
        )

    print("=" * 60)
    print("Aligned Date Range:")
    print("From:", common_dates[0])
    print("To  :", common_dates[-1])
    print("Total aligned days before feature shift:", len(common_dates))
    print("Total stocks:", len(keys))
    print("Fixed sample window is fully contained in aligned dates.")
    print("=" * 60)


    all_data = []
    feature_dates = None

    for k in keys:
        df = final_dict[k].loc[common_dates]
        feat = make_features(df)

        if feature_dates is None:
            feature_dates = pd.DatetimeIndex(feat.index)
        else:
            if not feature_dates.equals(pd.DatetimeIndex(feat.index)):
                raise RuntimeError(
                    f"Feature dates are not aligned for stock {k}. "
                    "This should not happen after common-date alignment."
                )

        all_data.append(feat.values)

    all_data = np.stack(all_data, axis=1)


    mean = all_data.mean(axis=1, keepdims=True)
    std = all_data.std(axis=1, keepdims=True)

    all_data = (all_data - mean) / (std + EPS)
    all_data = np.nan_to_num(all_data)

    print("=" * 60)
    print("Feature tensor:")
    print("Shape:", all_data.shape)
    print("Feature date range:")
    print("From:", feature_dates[0])
    print("To  :", feature_dates[-1])
    print("Feature dates:", len(feature_dates))
    print("=" * 60)

    return all_data, stock_names, feature_dates


def main():

    all_data, stock_names, dates = load_and_filter_data(DATA_FOLDER)

    np.save(OUTPUT_FILE, all_data)

    np.savez(
        META_FILE,
        stock_names=np.array(stock_names),
        dates=np.array(dates, dtype="datetime64[D]")
    )

    pd.DataFrame({"stock": stock_names}).to_csv(
        "selected_stocks.csv", index=False
    )

    print("Saved:")
    print("-", OUTPUT_FILE, all_data.shape)
    print("-", META_FILE)
    print("-", "selected_stocks.csv")


if __name__ == "__main__":
    main()
