import pandas as pd
import sqlite3
import re
from datetime import datetime

# --- Configuration ---
CSV_PATH = '/path/to/defined_IUIDs_template.csv'  # Update with your actual CSV file path
EXCEL_PATH = '/path/to/treatment_metadata.xlsx'  # Update with your actual Excel file path
SQLITE_PATH = '/path/to/dicom_metadata.sqlite3'  # Update with your actual SQLite DB file path
OUTPUT_CSV = '/path/to/defined_IUIDs_populated.csv'

# Mapping the dates from Excel to the specific CSV columns
DATE_COL_MAPPING = {
    'MRT Datum präop': ('preop_ax_km', 'preop_flair'),
    'MRT Datum früh postop': ('postop_ax_km', 'postop_flair'),
    'MRT Datum zur Bestrahlungsplanung': ('bestrahlung_ax_km', 'bestrahlung_flair'),
    'MRT (RN? Rezidiv)': ('pseudo_ax_km', 'pseudo_flair'),
    'MRT Datum zur Festlegung auf RN o. Rezidiv durch Tubo/Interdisziplinär': ('therapie_ax_km', 'therapie_flair'),
    'MRT Datum 1. Verlauf nach Therapie': ('posttherapie_ax_km', 'posttherapie_flair')
}


# --- Heuristic Scoring Functions ---

def score_t1_km(desc):
    """
    Scores a SeriesDescription for a T1 post-contrast (KM) axial scan.
    Aggressively avoids pre-contrast scans (praeKM, nativ, pre).
    """
    if not isinstance(desc, str): return -999
    desc_lower = desc.lower()
    score = 0

    # 1. The Pre-Contrast Nuke (Destroy the score if it's explicitly pre-contrast)
    pre_contrast_patterns = [r'praekm', r'prekm', r'präkm', r'\bpre\b', r'\bprae\b', r'\bprä\b', r'nativ', r'ohne\b']
    for pat in pre_contrast_patterns:
        if re.search(pat, desc_lower):
            score -= 1000  # Unrecoverable penalty

    # 2. General Negative Filters (Derived sequences, perfusions, recons)
    bad_patterns = [
        r'sub', r'adc', r'trace', r'dwi', r'tof', r'mip', r'scout', r'localizer',
        r'perf', r'moco', r'ttp', r'cbv', r'cbf', r'mtt', r'rgb', r'angio',
        r'b1000', r'b0', r'csi', r'spectro', r'svs', r'rekon', r'\brek\b', r'\brec\b'
    ]
    for pat in bad_patterns:
        if re.search(pat, desc_lower):
            score -= 500

    if any(k in desc_lower for k in ['t2', 'flair', 'stir', 'tirm']):
        score -= 200

    # 3. Length Penalty (Slight preference for shorter, base descriptions)
    score -= len(desc)

    # 4. The "Golden Ticket" (Absolute highest priority for ax km / axial km)
    if re.search(r'\b(ax|axial)\s*\+?km\b', desc_lower):
        score += 1000

    # 5. Must-have / highly-preferred Contrast Agent indicators
    # We reward 'km' only if it exists, knowing 'praekm' is already nuked above.
    if re.search(r'(km|\+km|\bpost\b|\+c|gad)', desc_lower):
        score += 200
    else:
        score -= 100  # Strongly penalize if no contrast is mentioned

    # 6. Scan Direction Preferences
    if re.search(r'(ax|tra|trans|axial)', desc_lower):
        score += 100
    elif 'cor' in desc_lower:
        score += 30
    elif 'sag' in desc_lower:
        score += 10

    # 7. T1 Specific Indicators
    if re.search(r'(t1|mprage|vibe|tfe|spgr)', desc_lower):
        score += 100

    return score


def score_flair(desc):
    """
    Scores a SeriesDescription for a T2 FLAIR axial scan.
    """
    if not isinstance(desc, str): return -999
    desc_lower = desc.lower()
    score = 0

    # 1. Negative Filters
    bad_patterns = [
        r'sub', r'adc', r'trace', r'dwi', r'tof', r'mip', r'scout', r'localizer',
        r'perf', r'moco', r'ttp', r'cbv', r'cbf', r'mtt', r'rgb', r'angio',
        r'b1000', r'b0', r'csi', r'spectro', r'svs', r'rekon', r'\brek\b', r'\brec\b'
    ]
    for pat in bad_patterns:
        if re.search(pat, desc_lower):
            score -= 500

    if re.search(r'(km|\+km|\bpost\b|\+c|gad)', desc_lower):
        score -= 100  # Penalize contrast on FLAIR

    # 2. Length Penalty (Slight preference for shorter, base descriptions)
    score -= len(desc)

    # 3. The "Golden Ticket" (Absolute highest priority for ax flair / axial flair)
    if re.search(r'\b(ax|axial)\s*flair\b', desc_lower):
        score += 1000

    # 4. FLAIR / Dark-Fluid indicators (Crucial)
    if re.search(r'(flair|dark.?fluid|tirm)', desc_lower):
        score += 200
    else:
        score -= 200  # Heavy penalty if it's not actually a FLAIR

    # 5. Scan Direction Preferences
    if re.search(r'(ax|tra|trans|axial)', desc_lower):
        score += 100
    elif 'cor' in desc_lower:
        score += 30
    elif 'sag' in desc_lower:
        score += 10

    # 6. T2 Indicators
    if 't2' in desc_lower:
        score += 50

    return score


# --- Main Processing ---

def process_data():
    df_csv = pd.read_csv(CSV_PATH)

    df_excel = pd.read_excel(EXCEL_PATH, header=1)
    df_excel.columns = [str(c).strip() for c in df_excel.columns]

    conn = sqlite3.connect(SQLITE_PATH)

    for index, csv_row in df_csv.iterrows():
        vid = csv_row['Verschluesselungs ID']

        excel_match = df_excel[df_excel['Verschlüsselungs ID'] == vid]
        if excel_match.empty:
            continue

        excel_row = excel_match.iloc[0]

        for date_col, (km_col, flair_col) in DATE_COL_MAPPING.items():
            raw_date = excel_row.get(date_col)

            if pd.isna(raw_date):
                continue

            try:
                if isinstance(raw_date, str):
                    dt_obj = datetime.strptime(raw_date.strip(), '%d.%m.%Y')
                else:
                    dt_obj = raw_date
                formatted_date = dt_obj.strftime('%Y%m%d')
            except Exception as e:
                print(f"Skipping unparseable date '{raw_date}' for patient {vid}: {e}")
                continue

            query = """
                    SELECT SeriesInstanceUID, SeriesDescription
                    FROM file_properties
                    WHERE PatientID = ? \
                      AND StudyDate = ? \
                    """
            series_df = pd.read_sql_query(query, conn, params=(vid, formatted_date))

            if series_df.empty:
                continue

            best_km_uid, best_km_desc, max_km_score = None, None, -float('inf')
            best_flair_uid, best_flair_desc, max_flair_score = None, None, -float('inf')

            for _, s_row in series_df.iterrows():
                uid = s_row['SeriesInstanceUID']
                desc = s_row['SeriesDescription']

                # Evaluate T1 KM
                km_score = score_t1_km(desc)
                if km_score > max_km_score and km_score > 0:
                    max_km_score = km_score
                    best_km_uid = uid
                    best_km_desc = desc

                # Evaluate FLAIR
                flair_score = score_flair(desc)
                if flair_score > max_flair_score and flair_score > 0:
                    max_flair_score = flair_score
                    best_flair_uid = uid
                    best_flair_desc = desc

            # Update the CSV DataFrame
            if best_km_uid:
                df_csv.at[index, km_col] = f"{best_km_uid} [{best_km_desc}]"

            if best_flair_uid:
                df_csv.at[index, flair_col] = f"{best_flair_uid} [{best_flair_desc}]"

    df_csv.to_csv(OUTPUT_CSV, index=False)
    conn.close()
    print(f"Done! Successfully populated the CSV and saved as '{OUTPUT_CSV}'.")


if __name__ == '__main__':
    process_data()