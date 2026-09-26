
# ============================================================
# AURORA VOLUME PROFILE SCANNER
# STANDALONE PRODUCTION SCANNER
# ============================================================

import os
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from io import StringIO


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PERIOD = "2y"

PROFILE_DAYS = 30
PROFILE_COUNT = 5
PROFILE_STEP = 5

BINS = 24
VALUE_AREA_PCT = 0.70

OUTPUT_DIR = "output"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# NIFTY 200 UNIVERSE
# ============================================================

def get_nifty200():

    url = (
        "https://www.niftyindices.com/"
        "IndexConstituent/ind_nifty200list.csv"
    )

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept":
            "text/csv,application/csv,text/plain,*/*"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    nifty200 = pd.read_csv(
        StringIO(response.text)
    )

    nifty200["Yahoo_Symbol"] = (
        nifty200["Symbol"]
        .astype(str)
        .str.strip()
        + ".NS"
    )

    return nifty200


# ============================================================
# DAILY DATA
# ============================================================

def download_stock_data(
    nifty200
):

    stock_data = {}
    failed_stocks = []

    for i, symbol in enumerate(
        nifty200["Yahoo_Symbol"],
        start=1
    ):

        try:

            df = yf.download(
                symbol,
                period=DATA_PERIOD,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False
            )

            if (
                df is None
                or df.empty
            ):

                failed_stocks.append(
                    symbol
                )

                continue

            if isinstance(
                df.columns,
                pd.MultiIndex
            ):

                df.columns = (
                    df.columns
                    .get_level_values(0)
                )

            required = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]

            if not all(
                column in df.columns
                for column in required
            ):

                failed_stocks.append(
                    symbol
                )

                continue

            df = (
                df[required]
                .dropna()
                .copy()
            )

            if len(df) < 50:

                failed_stocks.append(
                    symbol
                )

                continue

            stock_data[symbol] = df

        except Exception as error:

            print(
                f"ERROR {symbol}: {error}"
            )

            failed_stocks.append(
                symbol
            )

        time.sleep(0.05)

        if i % 25 == 0:

            print(
                f"Downloaded: "
                f"{i}/"
                f"{len(nifty200)}"
            )

    return (
        stock_data,
        failed_stocks
    )


# ============================================================
# EXACT V2 DAILY VOLUME PROFILE
# ============================================================

def calculate_daily_volume_profile_v2(
    df,
    bins=BINS,
    value_area_pct=VALUE_AREA_PCT
):

    data = df.dropna(
        subset=[
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    ).copy()

    if len(data) < 5:
        return None

    high = float(
        data["High"].max()
    )

    low = float(
        data["Low"].min()
    )

    if high <= low:
        return None

    edges = np.linspace(
        low,
        high,
        bins + 1
    )

    centres = (
        edges[:-1]
        +
        edges[1:]
    ) / 2.0

    volume_profile = np.zeros(
        bins
    )

    typical_price = (
        data["High"].astype(float)
        +
        data["Low"].astype(float)
        +
        data["Close"].astype(float)
    ) / 3.0

    volume = (
        data["Volume"]
        .astype(float)
    )

    bin_index = (
        np.digitize(
            typical_price,
            edges
        )
        - 1
    )

    bin_index = np.clip(
        bin_index,
        0,
        bins - 1
    )

    for idx, vol in zip(
        bin_index,
        volume
    ):

        volume_profile[idx] += vol

    total_volume = float(
        volume_profile.sum()
    )

    if total_volume <= 0:
        return None

    poc_index = int(
        np.argmax(
            volume_profile
        )
    )

    poc = float(
        centres[poc_index]
    )

    target_volume = (
        total_volume
        *
        value_area_pct
    )

    accumulated_volume = float(
        volume_profile[poc_index]
    )

    lower_index = poc_index
    upper_index = poc_index

    while (
        accumulated_volume
        <
        target_volume
    ):

        lower_volume = (
            volume_profile[
                lower_index - 1
            ]
            if lower_index > 0
            else -1
        )

        upper_volume = (
            volume_profile[
                upper_index + 1
            ]
            if upper_index < bins - 1
            else -1
        )

        if (
            lower_volume < 0
            and
            upper_volume < 0
        ):

            break

        if upper_volume >= lower_volume:

            if upper_index < bins - 1:

                upper_index += 1

                accumulated_volume += (
                    volume_profile[
                        upper_index
                    ]
                )

            else:

                lower_index -= 1

                accumulated_volume += (
                    volume_profile[
                        lower_index
                    ]
                )

        else:

            if lower_index > 0:

                lower_index -= 1

                accumulated_volume += (
                    volume_profile[
                        lower_index
                    ]
                )

            else:

                upper_index += 1

                accumulated_volume += (
                    volume_profile[
                        upper_index
                    ]
                )

    # IMPORTANT:
    # The validated V2 engine uses BIN CENTRES
    # for VAL and VAH.

    val = float(
        centres[lower_index]
    )

    vah = float(
        centres[upper_index]
    )

    return {
        "VAL": val,
        "POC": poc,
        "VAH": vah,
        "VA Width": vah - val,
        "VA Centre": (vah + val) / 2.0
    }


# ============================================================
# FIVE OVERLAPPING PROFILES
# ============================================================

def calculate_value_area_migration_v2(
    df
):

    data = df.dropna().copy()

    required_days = (
        PROFILE_DAYS
        +
        (
            PROFILE_COUNT - 1
        )
        *
        PROFILE_STEP
    )

    if len(data) < required_days:
        return None

    profiles = []

    for i in range(
        PROFILE_COUNT
    ):

        end_position = (
            len(data)
            -
            i * PROFILE_STEP
        )

        start_position = (
            end_position
            -
            PROFILE_DAYS
        )

        if start_position < 0:
            continue

        window = data.iloc[
            start_position:end_position
        ]

        profile = (
            calculate_daily_volume_profile_v2(
                window
            )
        )

        if profile is None:
            continue

        profiles.append({
            "Profile": i + 1,
            "VAL": profile["VAL"],
            "POC": profile["POC"],
            "VAH": profile["VAH"],
            "VA Width": profile["VA Width"],
            "VA Centre": profile["VA Centre"]
        })

    if len(profiles) < 5:
        return None

    return pd.DataFrame(
        profiles
    )


# ============================================================
# BUILD ALL FIVE PROFILES
# ============================================================

def build_all_profiles(
    stock_data
):

    all_profiles = []

    for symbol, df in stock_data.items():

        result = (
            calculate_value_area_migration_v2(
                df
            )
        )

        if result is None:
            continue

        result["Stock"] = (
            symbol.replace(
                ".NS",
                ""
            )
        )

        all_profiles.append(
            result
        )

    if not all_profiles:
        return pd.DataFrame()

    result = pd.concat(
        all_profiles,
        ignore_index=True
    )

    return (
        result[
            [
                "Stock",
                "Profile",
                "VAL",
                "POC",
                "VAH",
                "VA Width",
                "VA Centre"
            ]
        ]
        .sort_values(
            [
                "Stock",
                "Profile"
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# HISTORICAL STRUCTURAL TEST
# ============================================================

def build_daily_structure(
    profiles
):

    rows = []

    for stock, group in (
        profiles.groupby("Stock")
    ):

        group = (
            group
            .sort_values("Profile")
            .reset_index(drop=True)
        )

        if len(group) != 5:
            continue

        latest = group.iloc[0]
        oldest = group.iloc[4]

        def range_pct(series):

            mean_value = series.mean()

            if mean_value == 0:
                return np.nan

            return (
                (
                    series.max()
                    -
                    series.min()
                )
                /
                mean_value
                *
                100
            )

        val_range = range_pct(
            group["VAL"]
        )

        poc_range = range_pct(
            group["POC"]
        )

        vah_range = range_pct(
            group["VAH"]
        )

        centre_range = range_pct(
            group["VA Centre"]
        )

        centre_endpoint_move = (
            abs(
                latest["VA Centre"]
                -
                oldest["VA Centre"]
            )
            /
            abs(
                oldest["VA Centre"]
            )
            *
            100
        )

        centre_values = (
            group["VA Centre"]
            .to_numpy(
                dtype=float
            )
        )

        step_moves = np.abs(
            np.diff(
                centre_values
            )
        )

        centre_path = (
            step_moves.sum()
            /
            np.mean(
                centre_values
            )
            *
            100
        )

        if centre_path == 0:

            efficiency = 0.0

        else:

            efficiency = (
                centre_endpoint_move
                /
                centre_path
            )

        if (
            latest["VA Centre"]
            >
            oldest["VA Centre"]
        ):

            direction = "UP"

        elif (
            latest["VA Centre"]
            <
            oldest["VA Centre"]
        ):

            direction = "DOWN"

        else:

            direction = "FLAT"

        maximum_boundary_range = max(
            val_range,
            poc_range,
            vah_range
        )

        average_boundary_range = np.mean([
            val_range,
            poc_range,
            vah_range
        ])

        overall_envelope = (
            (
                max(
                    group["VAH"].max(),
                    group["POC"].max(),
                    group["VAL"].max()
                )
                -
                min(
                    group["VAH"].min(),
                    group["POC"].min(),
                    group["VAL"].min()
                )
            )
            /
            np.mean(
                group[
                    [
                        "VAL",
                        "POC",
                        "VAH"
                    ]
                ].stack()
            )
            *
            100
        )

        rows.append({

            "Stock": stock,

            "VAL Range %":
                val_range,

            "POC Range %":
                poc_range,

            "VAH Range %":
                vah_range,

            "Centre Range %":
                centre_range,

            "Average Boundary Range %":
                average_boundary_range,

            "Maximum Boundary Range %":
                maximum_boundary_range,

            "Centre Endpoint Move %":
                centre_endpoint_move,

            "Centre Path %":
                centre_path,

            "Directional Efficiency":
                efficiency,

            "Centre Direction":
                direction,

            "Overall VA Envelope %":
                overall_envelope
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# LATEST PROFILE vs PREVIOUS FOUR
# ============================================================

def build_latest_stability(
    profiles
):

    rows = []

    for stock, group in (
        profiles.groupby("Stock")
    ):

        group = (
            group
            .sort_values(
                "Profile",
                ascending=False
            )
            .reset_index(drop=True)
        )

        if len(group) != 5:
            continue

        previous = group.iloc[:4]
        latest = group.iloc[4]

        def position(
            latest_value,
            previous_values
        ):

            lower = previous_values.min()
            upper = previous_values.max()

            if (
                lower
                <=
                latest_value
                <=
                upper
            ):

                return (
                    "INSIDE",
                    0.0,
                    lower,
                    upper
                )

            if latest_value < lower:

                distance = (
                    (
                        lower
                        -
                        latest_value
                    )
                    /
                    abs(
                        previous_values.mean()
                    )
                    *
                    100
                )

                return (
                    "BELOW",
                    distance,
                    lower,
                    upper
                )

            distance = (
                (
                    latest_value
                    -
                    upper
                )
                /
                abs(
                    previous_values.mean()
                )
                *
                100
            )

            return (
                "ABOVE",
                distance,
                lower,
                upper
            )

        val = position(
            float(latest["VAL"]),
            previous["VAL"].astype(float)
        )

        poc = position(
            float(latest["POC"]),
            previous["POC"].astype(float)
        )

        vah = position(
            float(latest["VAH"]),
            previous["VAH"].astype(float)
        )

        centre = position(
            float(latest["VA Centre"]),
            previous["VA Centre"].astype(float)
        )

        inside_count = sum([
            val[0] == "INSIDE",
            poc[0] == "INSIDE",
            vah[0] == "INSIDE"
        ])

        previous_centre_mean = (
            previous["VA Centre"]
            .astype(float)
            .mean()
        )

        latest_centre = float(
            latest["VA Centre"]
        )

        centre_displacement = (
            abs(
                latest_centre
                -
                previous_centre_mean
            )
            /
            abs(
                previous_centre_mean
            )
            *
            100
        )

        rows.append({

            "Stock": stock,

            "Latest VAL":
                float(latest["VAL"]),

            "Latest POC":
                float(latest["POC"]),

            "Latest VAH":
                float(latest["VAH"]),

            "Latest VA Centre":
                latest_centre,

            "VAL Position":
                val[0],

            "POC Position":
                poc[0],

            "VAH Position":
                vah[0],

            "Centre Position":
                centre[0],

            "Boundaries Inside Previous Range":
                inside_count,

            "Latest Centre Displacement %":
                centre_displacement
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# CURRENT VALUE AREA
# ============================================================

def build_current_values(
    stock_data,
    profiles
):

    rows = []

    for symbol, df in stock_data.items():

        stock = symbol.replace(
            ".NS",
            ""
        )

        profile = profiles[
            (
                profiles["Stock"]
                ==
                stock
            )
            &
            (
                profiles["Profile"]
                ==
                1
            )
        ]

        if profile.empty:
            continue

        row = profile.iloc[0]

        current_price = float(
            df["Close"].iloc[-1]
        )

        rows.append({

            "Stock": stock,

            "Current Price":
                current_price,

            "VAL":
                row["VAL"],

            "POC":
                row["POC"],

            "VAH":
                row["VAH"],

            "VA Centre":
                row["VA Centre"],

            "VA Width":
                row["VA Width"],

            "Distance from VAL %":
                (
                    (
                        current_price
                        -
                        row["VAL"]
                    )
                    /
                    row["VAL"]
                    *
                    100
                ),

            "Distance from POC %":
                (
                    (
                        current_price
                        -
                        row["POC"]
                    )
                    /
                    row["POC"]
                    *
                    100
                ),

            "Distance from VAH %":
                (
                    (
                        current_price
                        -
                        row["VAH"]
                    )
                    /
                    row["VAH"]
                    *
                    100
                )
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN SCANNER
# ============================================================

def run_scanner():

    print("=" * 70)
    print("AURORA VOLUME PROFILE SCANNER")
    print("GITHUB PRODUCTION RUN")
    print("=" * 70)

    # --------------------------------------------------------
    # Universe
    # --------------------------------------------------------

    nifty200 = get_nifty200()

    print(
        f"\nNifty 200 stocks: "
        f"{len(nifty200)}"
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print(
        "\nDownloading daily OHLCV..."
    )

    stock_data, failed = (
        download_stock_data(
            nifty200
        )
    )

    print(
        f"\nSuccessful: "
        f"{len(stock_data)}"
    )

    print(
        f"Failed: "
        f"{len(failed)}"
    )

    # --------------------------------------------------------
    # Five profiles
    # --------------------------------------------------------

    print(
        "\nBuilding five profiles..."
    )

    profiles = build_all_profiles(
        stock_data
    )

    print(
        f"Profile records: "
        f"{len(profiles)}"
    )

    # --------------------------------------------------------
    # Historical structure
    # --------------------------------------------------------

    structure = build_daily_structure(
        profiles
    )

    val_reference = (
        structure[
            "VAL Range %"
        ].median()
    )

    poc_reference = (
        structure[
            "POC Range %"
        ].median()
    )

    vah_reference = (
        structure[
            "VAH Range %"
        ].median()
    )

    efficiency_reference = (
        structure[
            "Directional Efficiency"
        ].median()
    )

    structure[
        "VAL Contained"
    ] = (
        structure["VAL Range %"]
        <=
        val_reference
    )

    structure[
        "POC Contained"
    ] = (
        structure["POC Range %"]
        <=
        poc_reference
    )

    structure[
        "VAH Contained"
    ] = (
        structure["VAH Range %"]
        <=
        vah_reference
    )

    structure[
        "Rotational Path"
    ] = (
        structure[
            "Directional Efficiency"
        ]
        <=
        efficiency_reference
    )

    structure[
        "All Three Boundaries Contained"
    ] = (
        structure["VAL Contained"]
        &
        structure["POC Contained"]
        &
        structure["VAH Contained"]
    )

    structure[
        "Contained + Rotational"
    ] = (
        structure[
            "All Three Boundaries Contained"
        ]
        &
        structure[
            "Rotational Path"
        ]
    )

    # --------------------------------------------------------
    # Latest stability
    # --------------------------------------------------------

    latest = build_latest_stability(
        profiles
    )

    latest[
        "Latest Profile Stable"
    ] = (
        latest[
            "Boundaries Inside Previous Range"
        ]
        ==
        3
    )

    # --------------------------------------------------------
    # Current values
    # --------------------------------------------------------

    current = build_current_values(
        stock_data,
        profiles
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = pd.merge(
        structure,
        latest,
        on="Stock",
        how="inner"
    )

    combined = pd.merge(
        combined,
        current,
        on="Stock",
        how="left"
    )

    combined[
        "Both Tests Satisfied"
    ] = (
        combined[
            "Contained + Rotational"
        ]
        &
        combined[
            "Latest Profile Stable"
        ]
    )

    final = (
        combined[
            combined[
                "Both Tests Satisfied"
            ]
        ]
        .copy()
        .sort_values("Stock")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Capture timestamp
    # --------------------------------------------------------

    capture_time = datetime.now()

    capture_timestamp = (
        capture_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    # --------------------------------------------------------
    # Round display values
    # --------------------------------------------------------

    price_columns = [
        "Current Price",
        "VAL",
        "POC",
        "VAH",
        "VA Centre",
        "VA Width"
    ]

    percentage_columns = [
        "VAL Range %",
        "POC Range %",
        "VAH Range %",
        "Centre Range %",
        "Directional Efficiency",
        "Latest Centre Displacement %",
        "Distance from VAL %",
        "Distance from POC %",
        "Distance from VAH %"
    ]

    for column in price_columns:

        final[column] = (
            pd.to_numeric(
                final[column],
                errors="coerce"
            )
            .round(2)
        )

    for column in percentage_columns:

        final[column] = (
            pd.to_numeric(
                final[column],
                errors="coerce"
            )
            .round(2)
        )

    # --------------------------------------------------------
    # Save candidate CSV
    # --------------------------------------------------------

    final.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "aurora_volume_profile_candidates.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Save complete profiles
    # --------------------------------------------------------

    profiles.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "aurora_five_profiles.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {

        "scanner":
            "Aurora Volume Profile Scanner",

        "universe":
            "Nifty 200",

        "timeframe":
            "Daily",

        "profile_length_sessions":
            PROFILE_DAYS,

        "number_of_profiles":
            PROFILE_COUNT,

        "profile_shift_sessions":
            PROFILE_STEP,

        "bins":
            BINS,

        "value_area_percentage":
            VALUE_AREA_PCT,

        "stocks_processed":
            int(
                len(stock_data)
            ),

        "profiles_generated":
            int(
                len(profiles)
            ),

        "historical_candidates":
            int(
                structure[
                    "Contained + Rotational"
                ].sum()
            ),

        "latest_stable_candidates":
            int(
                latest[
                    "Latest Profile Stable"
                ].sum()
            ),

        "final_candidates":
            int(
                len(final)
            ),

        "capture_timestamp":
            capture_timestamp,

        "final_candidate_stocks":
            final[
                "Stock"
            ].tolist()
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "scan_summary.json"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "SCAN COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nCapture: "
        f"{capture_timestamp}"
    )

    print(
        f"Stocks processed: "
        f"{len(stock_data)}"
    )

    print(
        f"Profiles generated: "
        f"{len(profiles)}"
    )

    print(
        f"Historical candidates: "
        f"{summary['historical_candidates']}"
    )

    print(
        f"Latest stable: "
        f"{summary['latest_stable_candidates']}"
    )

    print(
        f"FINAL CANDIDATES: "
        f"{len(final)}"
    )

    print("\nCandidates:")

    print(
        final[
            [
                "Stock",
                "Current Price",
                "VAL",
                "POC",
                "VAH",
                "VA Centre",
                "VA Width"
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":

    run_scanner()
