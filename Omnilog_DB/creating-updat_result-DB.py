
# Importation of commonly used libraries
import pandas as pa
from pathlib import Path


# Appending the path to the python files for data processing, formatting, and visualization to the system path to be able to import them
import sys, os, re
dir_path = os.path.abspath("/home/elouanln/projects/def-jcomte/elouanln/Sandbox/Code/Phenotyping/Omnilog/analysis_pipeline/library")
if dir_path not in sys.path:
    sys.path.append(dir_path)

# Importation of python files for data processing, formatting, and visualization
import Data_plot as dp
import Growth_analysis as da

# Defining the path to the directory containing the data and the output directory for the aggregated data
general_dir = ""
output_dir = "/home/elouanln/scratch/Sandbox/DB_omnilog/"

# Dictionary to store plate plans
PM4_plan = {'A1':'Negative_Control', 'A2':'Sodium_Phosphate', 'A3':'Tetrasodium_pyrophosphate', 'A4':'Trimeta_Phosphate', 'A5':'Tripoly_Phosphate', 'A6':'Triethyl_Phosphate', 'A7':'Hypophosphite', 'A8':'Adenosine-2’-monophosphate', 'A9':'Adenosine-3’-monophosphate', 'A10':'Adenosine-5’-monophosphate', 'A11':'Adenosine-2’,3’-cyclic_monophosphate', 'A12':'Adenosine-3’,5’-cyclic_monophosphate', 'B1':'Thiophosphate', 'B2':'Dithiophosphate', 'B3':'D,L-α-Glycerol_Phosphate', 'B4':'ß-Glycerol_Phosphate', 'B5':'Carbamyl_Phosphate', 'B6':'D-2-Phospho-Glyceric_Acid', 'B7':'D-3-Phospho-Glyceric_Acid', 'B8':'Guanosine-2’-monophosphate', 'B9':'Guanosine-3’-monophosphate', 'B10':'Guanosine-_5’_-monophosphate', 'B11':'Guanosine-_2’,3’-cyclic_monophosphate', 'B12':'Guanosine-_3’,5’-cyclic_monophosphate', 'C1':'Phosphoenol_Pyruvate', 'C2':'Phospho-_Glycolic_Acid', 'C3':'D-Glucose-1-Phosphate', 'C4':'D-Glucose-6-Phosphate', 'C5':'2-Deoxy-D-Glucose-6-Phosphate', 'C6':'D-Glucosamine-6-Phosphate', 'C7':'6-Phospho-Gluconic_Acid', 'C8':'Cytidine-2-monophosphate', 'C9':'Cytidine-3-monophosphate', 'C10':'Cytidine-5’-monophosphate', 'C11':'Cytidine-2’,3’-cyclic_monophosphate', 'C12':'Cytidine-3’,5’-cyclic_monophosphate', 'D1':'D-Mannose-1-Phosphate', 'D2':'D-Mannose-6-Phosphate', 'D3':'Cysteamine-S-Phosphate', 'D4':'Phospho-L-Arginine', 'D5':'O-Phospho-D-Serine', 'D6':'O-Phospho-L-Serine', 'D7':'O-Phospho-L-Threonine', 'D8':'Uridine-2’-monophosphate', 'D9':'Uridine-3’-monophosphate', 'D10':'Uridine-5’-monophosphate', 'D11':'Uridine-2’,3’-_cyclic_monophosphate', 'D12':'Uridine-3’,5’-_cyclic_monophosphate', 'E1':'O-Phospho-D-Tyrosine', 'E2':'O-Phospho-L-Tyrosine', 'E3':'Phosphocreatine', 'E4':'Phosphocholine_chloride', 'E5':'O-Phosphoryl-Ethanolamine', 'E6':'Phosphono_Acetic_Acid', 'E7':'2-Aminoethyl_Phosphonic_Acid', 'E8':'Methylene_Diphosphonic_Acid', 'E9':'Thymidine-3’-monophosphate', 'E10':'Thymidine-5’-monophosphate', 'E11':'Inositol_Hexaphosphate', 'E12':'Thymidine_3’,5’-cyclic_monophosphate', 'F1':'Negative_Control', 'F2':'Sodium_Sulfate', 'F3':'Sodium_thiophosphate', 'F4':'Tetrathionate', 'F5':'Thiophosphate', 'F6':'Dithiophosphate', 'F7':'L-Cysteine', 'F8':'D-Cysteine', 'F9':'L-Cysteinyl-Glycine', 'F10':'L-Cysteic_Acid', 'F11':'Cysteamine', 'F12':'L-Cysteine_Sulfinic_Acid', 'G1':'N-Acetyl-L-Cysteine', 'G2':'S-Methyl-L-Cysteine', 'G3':'Cystathionine', 'G4':'Lanthionine', 'G5':'Glutathione', 'G6':'D,L-Ethionine', 'G7':'L-Methionine', 'G8':'D-Methionine', 'G9':'Glycyl-L-Methionine', 'G10':'N-Acetyl-D,L-Methionine', 'G11':'L-Methionine_Sulfoxide', 'G12':'L-Methionine_Sulfone', 'H1':'L-Djenkolic_Acid', 'H2':'Thiourea', 'H3':'1-Thio-ß-D-_Glucose', 'H4':'D,L-Lipoamide', 'H5':'Taurocholic_Acid', 'H6':'Taurine', 'H7':'Hypotaurine', 'H8':'m-Amino_benzene_sulfonic_acid', 'H9':'Butane_Sulfonic_Acid', 'H10':'2-Hydroxyethane_Sulfonic_Acid', 'H11':'Methane_Sulfonic_Acid', 'H12':'Tetramethylene_Sulfone'}
PM3_plan = {'A1':'Negative_Control', 'A2':'Ammonium_Formate', 'A3':'SodiumNitrite', 'A4':'SodiumNitrate', 'A5':'Urea', 'A6':'Biuret', 'A7':'L-Alanine', 'A8':'L-Arginine', 'A9':'L-Asparagine', 'A10':'L-Aspartic_Acid', 'A11':'L-Cysteine', 'A12':'L-Glutamic_Acid', 'B1':'L-Glutamine', 'B2':'Glycine', 'B3':'L-Histidine', 'B4':'L-Isoleucine', 'B5':'L-Leucine', 'B6':'L-Lysine', 'B7':'L-Methionine', 'B8':'L-Phenylalanine', 'B9':'L-Proline', 'B10':'L-Serine', 'B11':'L-Threonine', 'B12':'L-Tryptophan', 'C1':'L-Tyrosine', 'C2':'L-Valine', 'C3':'D-Alanine', 'C4':'D-Asparagine', 'C5':'D-Aspartic_Acid', 'C6':'D-Glutamic_Acid', 'C7':'D-Lysine', 'C8':'D-Serine', 'C9':'D-Valine', 'C10':'L-Citrulline', 'C11':'L-Homoserine', 'C12':'L-Ornithine', 'D1':'N-Acetyl-L-Glutamic_Acid', 'D2':'N-Phthaloyl-L-Glutamic_Acid', 'D3':'L-Pyroglutamic_Acid', 'D4':'Hydroxylamine', 'D5':'Methylamine', 'D6':'N-Amylamine', 'D7':'N-Butylamine', 'D8':'Ethylamine', 'D9':'Ethanolamine', 'D10':'Ethylenediamine', 'D11':'Putrescine', 'D12':'Agmatine', 'E1':'Histamine', 'E2':'ß-Phenylethyl-amine', 'E3':'Tyramine', 'E4':'Acetamide', 'E5':'Formamide', 'E6':'Glucuronamide', 'E7':'D,L-Lactamide', 'E8':'D-Glucosamine', 'E9':'D-Galactosamine', 'E10':'D-Mannosamine', 'E11':'N-Acetyl-D-Glucosamine', 'E12':'N-Acetyl-D-Galactosamine', 'F1':'N-Acetyl-D-Mannosamine', 'F2':'Adenine', 'F3':'Adenosine', 'F4':'Cytidine', 'F5':'Cytosine', 'F6':'Guanine', 'F7':'Guanosine', 'F8':'Thymine', 'F9':'Thymidine', 'F10':'Uracil', 'F11':'Uridine', 'F12':'Inosine', 'G1':'Xanthine', 'G2':'Xanthosine', 'G3':'Uric_Acid', 'G4':'Alloxan', 'G5':'Allantoin', 'G6':'Parabanic_Acid', 'G7':'D,L-α-Amino-N-Butyric_Acid', 'G8':'γ-Amino-N-_Butyric_Acid', 'G9':'ε-Amino-N-Caproic_Acid', 'G10':'D,L-α-Amino-Caprylic_Acid', 'G11':'δ-Amino-N-_Valeric_Acid', 'G12':'α-Amino-N-_Valeric_Acid', 'H1':'Ala-Asp', 'H2':'Ala-Gln', 'H3':'Ala-Glu', 'H4':'Ala-Gly', 'H5':'Ala-His', 'H6':'Ala-Leu', 'H7':'Ala-Thr', 'H8':'Gly-Asn', 'H9':'Gly-Gln', 'H10':'Gly-Glu', 'H11':'Gly-Met', 'H12':'Met-Ala'}

# CSV name example : Biolog_Assay_PM-3_Replicate-1_AC_02_P3C02-1_D0-H0.csv
# Regex pattern to extract the relevant information from the CSV file names
PAT = re.compile(
    r"^Biolog_[Aa]ssay_"
    r"(?P<plate_type>PM-\d+)_"
    r"Replicate-(?P<replicate>\d+)_"
    r"(?P<strain>[A-Za-z0-9]+_\d+)_"
    r"(?P<plate_serial>P\d[A-Z]{1,3}\d{2}-\d+)[_\-]"
    r"(?P<suffix>[DH]\d+-[DH]\d+|[DH]\d+)"
    r"\.csv$"
)
# dir to ignore when scanning for files
IGNORE = {".git", "archives", "backup", "__pycache__"}

# function to create the db
def scan(root_dir):
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"{root_dir} n'existe pas")
    ignores=[]
    List_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in IGNORE]  # Trimming the list of directories to ignore
        for name in filenames:
            m = PAT.match(name)
            if not m:
                if name.endswith(".csv"):
                    ignores.append(name)
                continue
            p = Path(dirpath) / name # path to the directory containing actual files
            List_files.append((str(p.resolve()), m.groupdict()))
            List_serial = m.group('plate_serial')
    print(f"Treated serial : {set(List_serial)}")
    print(f"Found {ignores} files to ignore in {root_dir}")
    print(f"Found {List_files} files in {root_dir}")
    return List_files

def associating_substrate(df):
    List_rows = df['Row_plate_name'].unique()
    List_columns = df['Columns_plate_name'].unique()
    Plate_types = df['plate_type'].unique()[0]
    for row in List_rows:
        for col in List_columns:
            if Plate_types == 'PM-3':
                substrate = PM3_plan[f"{row}{col}"]
            elif Plate_types == 'PM-4':
                substrate = PM4_plan[f"{row}{col}"]
            else:
                raise ValueError(f"Unknown plate type: {Plate_types}")
            df.loc[(df['Row_plate_name'] == row) & (df['Columns_plate_name'] == col), 'substrate'] = substrate
    return df

def aggregate_data(List_files): # A list containing tuple (path, regex_match) is expected as input
    data_collection = pa.DataFrame()
    metadata = dict()
    df_tmp = pa.DataFrame()

    # Assessing possible errors in the input list of files
    if not List_files:
        raise ValueError("Aucun fichier à agréger.")

    # collecting data in tmp df then concatenating them in the main df
    for file_path, labels in List_files:
        data_set, metadata[os.path.basename(file_path)] = dp.load_data(file_path)
        df_tmp = dp.data_formating(data_set, metadata[os.path.basename(file_path)])
        df_tmp['Path_to_file'] = file_path
        df_tmp['plate_type'] = labels['plate_type']
        df_tmp['strain'] = labels['strain']
        df_tmp['plate_serial'] = labels['plate_serial']
        df_tmp= associating_substrate(df_tmp)

        data_collection = pa.concat([data_collection, df_tmp], ignore_index=True)

    # Adding element information to the dataframe
    data_collection.loc[data_collection['plate_type'] == 'PM-3', 'Element'] = 'N'
    data_collection.loc[data_collection['plate_type'] == 'PM-4', 'Element'] = 'P'
    data_collection.loc[(data_collection['plate_type'] == 'PM-4') & (data_collection['Row_plate_name'].isin(['F', 'G', 'H'])), 'Element'] = 'S' # replace row F, G, H of PM-4 by S for sulfur

    orphan = data_collection.loc[data_collection['Element'].isna(), 'plate_type'].unique()
    if len(orphan):
        print(f"! plate_type sans Element assigné : {orphan} !")
    return data_collection, metadata

### Execution of the functions 
from datetime import datetime
import json

archives_dir = os.path.join(output_dir, "archives")
os.makedirs(archives_dir, exist_ok=True)

for file in os.listdir(output_dir):
    if os.path.isfile(os.path.join(output_dir, file)):
        os.replace(os.path.join(output_dir, file), os.path.join(archives_dir, file))


List_files = scan(general_dir)
data_collection, metadata = aggregate_data(List_files)
List_plate_types = sorted(data_collection['plate_type'].unique())
current_date = datetime.now().strftime("%Y%m%d_%H%M%S")

if not data_collection.empty and metadata:
    data_collection.to_csv(os.path.join(output_dir, f"DB_Omnilog_analysis_{'_'.join(List_plate_types)}_{current_date}.csv"), index=False)
    with open(os.path.join(output_dir, f"metadata_{'_'.join(List_plate_types)}_{current_date}.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
else:
    raise ValueError("No data found to create DB.")