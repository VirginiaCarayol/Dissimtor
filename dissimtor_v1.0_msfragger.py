# -*- coding: utf-8 -*-

import argparse
from strenum import StrEnum
import sys

# List of HLA-I alleles for which Dissimtor currently works.
ALLELES = ["HLA_A0101", "HLA_A0201", "HLA_A0202", "HLA_A0203", "HLA_A0204", 
           "HLA_A0205", "HLA_A0206", "HLA_A0207", "HLA_A0211", "HLA_A0301", 
           "HLA_A1101", "HLA_A1102", "HLA_A2301", "HLA_A2402", "HLA_A2501", 
           "HLA_A2601", "HLA_A2902", "HLA_A3001", "HLA_A3002", "HLA_A3201", 
           "HLA_A3301", "HLA_A3303", "HLA_A3401", "HLA_A3402", "HLA_A6601", 
           "HLA_A6801", "HLA_A6802", "HLA_A7401", "HLA_B0702", "HLA_B0801", 
           "HLA_B1301", "HLA_B1302", "HLA_B1402", "HLA_B1501", "HLA_B1502", 
           "HLA_B1503", "HLA_B1510", "HLA_B1517", "HLA_B1801", "HLA_B2701", 
           "HLA_B2702", "HLA_B2705", "HLA_B2707", "HLA_B2709", "HLA_B3501", 
           "HLA_B3502", "HLA_B3503", "HLA_B3507", "HLA_B3801", "HLA_B3802", 
           "HLA_B3901", "HLA_B4001", "HLA_B4002", "HLA_B4006", "HLA_B4101", 
           "HLA_B4201", "HLA_B4402", "HLA_B4403", "HLA_B4501", "HLA_B4601", 
           "HLA_B4901", "HLA_B5001", "HLA_B5101", "HLA_B5301", "HLA_B5502", 
           "HLA_B5601", "HLA_B5701", "HLA_B5703", "HLA_B5801", "HLA_C0102", 
           "HLA_C0202", "HLA_C0303", "HLA_C0304", "HLA_C0401", "HLA_C0501", 
           "HLA_C0602", "HLA_C0701", "HLA_C0702", "HLA_C0801", "HLA_C0802", 
           "HLA_C1202", "HLA_C1203", "HLA_C1402", "HLA_C1403", "HLA_C1502", 
           "HLA_C1601", "HLA_C1701", "H2_Db", "H2_Dd", "H2_Kb", "H2_Kd"] 

class FragpipeVersion(StrEnum):
    """Supported FragPipe versions."""
    V19 = "V19"
    V20 = "V20"
    V21 = "V21"
    V22 = "V22"

parser = argparse.ArgumentParser(prog='Dissimtor',description='Mass spectrometry (MS) identifications rescoring using artificial neural networks (ANN).')
parser.add_argument('--list-alleles', help="(optional) HLA-I alleles for which Dissimtor currently works.", action="store_true", dest="list_alleles")
parser.add_argument('--A1mol', type=str, help="HLA-I allele 1; write it in the form 'HLA_X0000' (for example, 'HLA_A0101'). To see the HLA-I alleles for which Dissimtor currently works, use the argument '--list-alleles'. If the rest of HLA-I alleles are not specified, they will be completed with this allele.")
parser.add_argument('--A2mol', type=str, help="(optional) HLA-I allele 2.")
parser.add_argument('--B1mol', type=str, help="(optional) HLA-I allele 3.")
parser.add_argument('--B2mol', type=str, help="(optional) HLA-I allele 4.")
parser.add_argument('--C1mol', type=str, help="(optional) HLA-I allele 5.")
parser.add_argument('--C2mol', type=str, help="(optional) HLA-I allele 6.")
parser.add_argument('--input-file', type=str, help="Input file name. Indicate the directory and name of the input file (with .pin extension; it is the output of MSFragger (FragPipe) with FDR 100%%).", dest="input_file")
parser.add_argument('--output-file', type=str, help="Output file name. Indicate the directory and name of the output file of Dissimtor (with .pin extension; it will be saved in .pin, .csv and .xlsx).", dest="output_file")
parser.add_argument('--fragpipe-version', type=FragpipeVersion, help=f"Version of FragPipe used to get the input_file. Versions supported: {', '.join(list(FragpipeVersion))}", choices=list(FragpipeVersion), dest="fragpipe_version")

args = parser.parse_args()

if args.list_alleles:
    print("Available alleles: ")
    for allele in ALLELES:
        print(f"  {allele}")
    sys.exit()

a1mol = args.A1mol
a2mol = args.A2mol or a1mol
b1mol = args.B1mol or a1mol
b2mol = args.B2mol or a1mol
c1mol = args.C1mol or a1mol
c2mol = args.C2mol or a1mol
input_file = args.input_file
output_file = args.output_file
fragpipe_version = args.fragpipe_version

if a1mol is None or input_file is None or output_file is None or fragpipe_version is None:
    print("The following arguments are required: --A1mol, --input-file, --output-file and --fragpipe-version.")
    sys.exit(1)

if not input_file.endswith('.pin'):
    print("Input file must be of type .pin")
    sys.exit(1)

if not output_file.endswith('.pin'):
    print("Output file must be of type .pin")
    sys.exit(1)

print("Importing modules...")
import numpy as np
import pandas as pd
import re
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings

warnings.filterwarnings('ignore')

# Set this environment variable to silence these warnings from tensorflow:
#
#   oneDNN custom operations are on. You may see slightly different numerical 
#   results due to floating-point round-off errors from different computation
#   orders. To turn them off, set the environment variable
#   `TF_ENABLE_ONEDNN_OPTS = 0`. 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

# Check existence and load the artificial neural network (ANN) models for the specified HLA-I alleles.

# Read the CSV file containing equivalences between HLA-I alleles and the closest available ANN models.
# If an exact ANN is not available for a given allele, the ANN of the most similar allele is used.
dfANN = pd.read_csv(os.path.join("ANN_alleles", "hla_ann_equivalence.csv"))

def get_ann_value(mol_name, mol_value):
    """Checks whether the ANN is available for the user-specified allele. If not, uses the most similar available allele instead."""
    if mol_value in dfANN['allele_short'].values:
        ann = dfANN.loc[dfANN['allele_short'] == mol_value, 'ANN'].values[0]
        if ann == mol_value:
            print(f"ANN available for {mol_value}")
        else:
            print(f"{ann} used instead of {mol_value} (see list of alleles for which Dissimtor has ANN with --list_alleles).")
        return ann
    else:
        print(f"{mol_name} unknown. Please enter a valid allele (write it in the form 'HLA_X0000' (for example, 'HLA_A0101')).")
        sys.exit(1)

a1mol = get_ann_value("a1mol", a1mol)
a2mol = get_ann_value("a2mol", a2mol)
b1mol = get_ann_value("b1mol", b1mol)
b2mol = get_ann_value("b2mol", b2mol)
c1mol = get_ann_value("c1mol", c1mol)
c2mol = get_ann_value("c2mol", c2mol)

def load_model(allele):
    """Loads an ANN for the specified allele."""
    model = os.path.join("ANN_alleles", f"dissimtor_{allele}.keras")
    if not os.path.exists(model):
        print(f"Error: artificial neural network for allele '{allele}' is not available.")
        sys.exit(1)
    else:
        return tf.keras.models.load_model(model)

print("Importing artificial neural networks...")
a1_ann = load_model(a1mol)
a2_ann = load_model(a2mol)
b1_ann = load_model(b1mol)
b2_ann = load_model(b2mol)
c1_ann = load_model(c1mol)
c2_ann = load_model(c2mol)

print("Doing some modifications to the data...")

# The .pin file is read and saved in the 'lines' variable.
with open(input_file, 'r') as f:
    lines = f.readlines()

# Check the version of FragPipe:
if fragpipe_version == FragpipeVersion.V19:
    # Cleaning the .pin file.
    input_file_clean = input_file + 'clean.pin'    

    cleaned_lines = []
    cleaned_lines.append(lines[0])  # Add the first line as is (column labels).

    for line in lines[1:]:
        fields = line.split('\t')
        # Check if the .pin file contains more than 28 columns.
        if len(fields) > 28:
            # Concatenate the extra columns starting from 28 to the 28th column
            extra_fields = ';'.join(fields[27:])
            fields[27] = fields[27] + ';' + extra_fields  # Concatenate extra fields to column 28
            cleaned_line = '\t'.join(fields[:28])  # Only keep up to column 28 (now with extra fields)
        else:
            cleaned_line = '\t'.join(fields)
        cleaned_lines.append(cleaned_line)

    # Save the processed lines to the file 'clean.pin'.
    with open(input_file_clean, 'w') as f:
        f.writelines(cleaned_lines)
    
    pept_0 = pd.read_csv(input_file_clean, sep="\t", header=0)

# In FragPipe version 20, 21 and 22:
if fragpipe_version == FragpipeVersion.V20 or fragpipe_version == FragpipeVersion.V21 or fragpipe_version == FragpipeVersion.V22:
    pept_0 = pd.read_csv(input_file, sep="\t")
    # Remove the number that corresponds to the charge of the peptide.
    # Each peptide is a string with parts separated by dots. The penultimate
    # part can end with a number, which corresponds to the charge. We are
    # removing this number on each peptide below.
    sliced_peptides = pept_0["Peptide"].str.split('.', expand=False)
    fixed_peptides = []

    for sliced_peptide in sliced_peptides:
        index = len(sliced_peptide)-2
        if sliced_peptide[index][-1].isdigit():
            sliced_peptide[index] = sliced_peptide[index][:-1]
        fixed_peptides.append(".".join(sliced_peptide))

    pept_0["Peptide"] = fixed_peptides

# Dissimtor currently only considers the modifications: oxidation of methionine, 
# acetylation (N-term) and carbamidomethylation of cysteine. A .pin document with
# other modifications should not be passed as --input-file. 
# Rows whose peptides contain selenocysteine ​​(U) are removed.
pept_0 = pept_0.loc[['U' not in p for p in pept_0['Peptide']]]

# Keep a copy of the 'Peptide' column like 'Sequence' (it will be used in the output).
pept_0['Sequence'] = pept_0['Peptide'] 
# Oxidation of methionine.
pept_0['Peptide'] = pept_0['Peptide'].str.replace('\[15.9949\]','')
# Acetylation (N-term).
pept_0['Peptide'] = pept_0['Peptide'].str.replace('n\[42.0106\]','N') 
# Carbamidomethylation of cysteine.
pept_0['Peptide'] = pept_0['Peptide'].str.replace('\[57.0215\]','') 
# The dots separate the flanking amino acids (before and after) from the identified 
# peptide.
pept_0['Peptide'] = pept_0['Peptide'].str.replace('\.','x') 

def strapplyc(values, pattern):
    """Function to apply regular expression patterns to a list."""
    result = []
    for value in values:
        match = re.search(pattern, value)
        if match:
            result.append(match.group(1))
    return result

# Apply the strapplyc() function to the ‘Peptide’ column: searches for 
# any string that starts and ends with "x", capturing everything in between (the 
# identified peptide).
pept_0['Peptide'] = strapplyc(pept_0['Peptide'], "x(.*)x")

# If --input-file has been provided with other modifications that Dissimtor does 
# not support, an error will be displayed.
# Create a set of valid amino acids:
aas = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", 
       "S", "T", "W", "Y", "V", "X"]
valid_aas = set(aas)

def contains_only_valid_chars(peptide):
    """Function to check if a peptide contains only valid characters (amino acids)."""
    return all(char in valid_aas for char in peptide)

pept_0['is_valid'] = pept_0['Peptide'].apply(contains_only_valid_chars)

# Peptides that do not satisfy these characteristics are shown.
if not pept_0['is_valid'].all() == True: # 'not pept_0['is_valid'].all()' will be 'True' if there is at least one 'False'.
    false_rows = pept_0[pept_0['is_valid'] == False]
    print("Error: Dissimtor currently only considers the modifications: oxidation of methionine, acetylation (N-term) and carbamidomethylation of cysteine. A .pin document with other modifications should not be passed as --input-file. Peptides that do not satisfy these characteristics are:")
    print(false_rows['Peptide'].values[0])
    sys.exit(1)

# Select some columns.
pept = pept_0[["ScanNr", "hyperscore", "Label", "Peptide"]]

# List of peptide lengths.
lengths = pept['Peptide'].apply(lambda x: len(x)).tolist()

# Dissimtor currently works with peptides of 8-11 amino acids, but it uses 9 
# amino acid peptides to calculate the affinity with HLA-I alleles.

# 8 amino acid peptides:
filters = [length == 8 for length in lengths]
pept8mer = pept.loc[filters]

# Add a random amino acid, 'X', in all possible positions.
pept8mer['nmer1'] = [s[:8] + 'X' for s in pept8mer['Peptide']]
pept8mer['nmer2'] = [s[:7] + 'X' + s[7:] for s in pept8mer['Peptide']]
pept8mer['nmer3'] = [s[:6] + 'X' + s[6:] for s in pept8mer['Peptide']]
pept8mer['nmer4'] = [s[:5] + 'X' + s[5:] for s in pept8mer['Peptide']]
pept8mer['nmer5'] = [s[:4] + 'X' + s[4:] for s in pept8mer['Peptide']]
pept8mer['nmer6'] = [s[:3] + 'X' + s[3:] for s in pept8mer['Peptide']]
pept8mer['nmer7'] = [s[:2] + 'X' + s[2:] for s in pept8mer['Peptide']]
pept8mer['nmer8'] = [s[:1] + 'X' + s[1:] for s in pept8mer['Peptide']]
pept8mer['nmer9'] = ['X' + s for s in pept8mer['Peptide']]

pept8mer = pept8mer.reset_index()
pept8mer = pd.melt(pept8mer, id_vars=["ScanNr", "hyperscore", "Label", "Peptide"], 
                 value_vars=['nmer1', 'nmer2', 'nmer3','nmer4','nmer5','nmer6','nmer7','nmer8','nmer9'])

# Peptides of 9-11 amino acids: take the possible peptides of 9 amino acids.
filters = [length > 8 for length in lengths]
pept9_11mer = pept.loc[filters]
pept9_11mer['nmer1'] = [s[:9] for s in pept9_11mer['Peptide']]
pept9_11mer['nmer2'] = [s[1:10] for s in pept9_11mer['Peptide']]
pept9_11mer['nmer3'] = [s[2:11] for s in pept9_11mer['Peptide']]

pept9_11mer = pept9_11mer.reset_index()
pept9_11mer = pd.melt(pept9_11mer, id_vars=["ScanNr", "hyperscore", "Label", "Peptide"], 
             value_vars=['nmer1', 'nmer2', 'nmer3'])

lengths1 = pept9_11mer['value'].apply(lambda x: len(x)).tolist()
filters1 = [length == 9 for length in lengths1]
pept9_11mer = pept9_11mer.loc[filters1]

# Peptides of 9 amino acids (9nmers-cores).
pept9mer = pd.concat([pept8mer,pept9_11mer])

# Embedding 9nmers-cores from MS results: amino acid transformation using BLOSUM50.
A = "5,-2,-1,-2,-1,-1,-1,0,-2,-1,-2,-1,-1,-3,-1,1,0,-3,-2,0,"
R = "-2,7,-1,-2,-4,1,0,-3,0,-4,-3,3,-2,-3,-3,-1,-1,-3,-1,-3,"
N = "-1,-1,7,2,-2,0,0,0,1,-3,-4,0,-2,-4,-2,1,0,-4,-2,-3,"
D = "-2,-2,2,8,-4,0,2,-1,-1,-4,-4,-1,-4,-5,-1,0,-1,-5,-3,-4,"
C = "-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1,"
Q = "-1,1,0,0,-3,7,2,-2,1,-3,-2,2,0,-4,-1,0,-1,-1,-1,-3,"
E = "-1,0,0,2,-3,2,6,-3,0,-4,-3,1,-2,-3,-1,-1,-1,-3,-2,-3,"
G = "0,-3,0,-1,-3,-2,-3,8,-2,-4,-4,-2,-3,-4,-2,0,-2,-3,-3,-4,"
H = "-2,0,1,-1,-3,1,0,-2,10,-4,-3,0,-1,-1,-2,-1,-2,-3,2,-4,"
I = "-1,-4,-3,-4,-2,-3,-4,-4,-4,5,2,-3,2,0,-3,-3,-1,-3,-1,4,"
L = "-2,-3,-4,-4,-2,-2,-3,-4,-3,2,5,-3,3,1,-4,-3,-1,-2,-1,1,"
K = "-1,3,0,-1,-3,2,1,-2,0,-3,-3,6,-2,-4,-1,0,-1,-3,-2,-3,"
M = "-1,-2,-2,-4,-2,0,-2,-3,-1,2,3,-2,7,0,-3,-2,-1,-1,0,1,"
F = "-3,-3,-4,-5,-2,-4,-3,-4,-1,0,1,-4,0,8,-4,-3,-2,1,4,-1,"
P = "-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3,"
S = "1,-1,1,0,-1,0,-1,0,-1,-3,-3,0,-2,-3,-1,5,2,-4,-2,-2,"
T = "0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,2,5,-3,-2,0,"
W = "-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1,1,-4,-4,-3,15,2,-3,"
Y = "-2,-1,-2,-3,-3,-1,-2,-3,2,-1,-1,-2,0,4,-3,-2,-2,2,8,-1,"
V = "0,-3,-3,-4,-1,-3,-3,-4,-4,4,1,-3,1,-1,-3,-2,0,-3,-1,5,"
X = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"

for (character, aminoacid) in [("A", A),
                               ("R", R),
                               ("N", N),
                               ("D", D),
                               ("C", C),
                               ("Q", Q),
                               ("E", E),
                               ("G", G),
                               ("H", H),
                               ("I", I),
                               ("L", L),
                               ("K", K),
                               ("M", M),
                               ("F", F),
                               ("P", P),
                               ("S", S),
                               ("T", T),
                               ("W", W),
                               ("Y", Y),
                               ("V", V),
                               ("X", X)]:
   pept9mer['value'] = pept9mer['value'].str.replace(character, aminoacid)

# Remake the dataframe so each amino acid has it's values split into separate columns.
pept_blosum50 = pept9mer['value'].str.split(',', expand=True) 
# Rename the columns by prefixing them with "P".
pept_blosum50.columns = [f"P{i}" for i in pept_blosum50.columns] 
# Remove the last column because it's always empty. This is because the amino acids
# have trailing commas.
pept_blosum50 = pept_blosum50.drop(columns=['P180']) 
# Convert values to numeric.
pept_blosum50 = pept_blosum50.apply(pd.to_numeric) 

print("Calculating the affinity of peptides for each specified HLA-I allele using their artificial neural network (Dissimtor score or 'DisScore')...")

# Affinity of peptides for each specified HLA-I allele using their ANN (ANNs have been loaded
# previously).
predA1 = (a1_ann.predict(pept_blosum50))
predA2 = (a2_ann.predict(pept_blosum50))
predB1 = (b1_ann.predict(pept_blosum50))
predB2 = (b2_ann.predict(pept_blosum50))
predC1 = (c1_ann.predict(pept_blosum50))
predC2 = (c2_ann.predict(pept_blosum50))

# Keep the maximum affinity value of each peptide: 'DisScore'.
pred_max = np.concatenate(list(map(max, predA1, predA2, predB1, predB2, predC1, predC2))) 
pept9mer['DisScore'] = pred_max
pept9mer_subset = pept9mer[['Peptide','DisScore']]
# Make sure to have only one row per peptide: we eliminate possible duplicate peptides, 
# keeping the highest 'DisScore' value.
pept9mer_subset = pept9mer_subset.sort_values('DisScore').groupby('Peptide',as_index=False).last()
pept9mer_subset = pept9mer_subset.drop_duplicates()
pept_final = pd.merge(pept_0, pept9mer_subset, on='Peptide')
# Use the copy of the 'Peptide' column unmodified (saved like 'Sequence').
pept_final['Peptide'] = pept_final['Sequence']
pept_final = pept_final.drop(columns=['Sequence', 'is_valid'])
pept_final = pept_final.drop_duplicates()
# Rearrange the columns (Percolator will use the DisScore column).
final_cols = ['DisScore', 'Peptide', 'Proteins']
other_cols = [col for col in pept_final.columns if col not in final_cols]
pept_final = pept_final[other_cols + final_cols]

print("Saving the results...")

# Save the results in .pin, .csv and .xlsx.
pept_final['Proteins'] = pept_final['Proteins'].str.replace(" ","\t")

csv_output_file = output_file[0 : (len(output_file) - len('.pin'))] + '.csv'
xlsx_output_file = output_file[0 : (len(output_file) - len('.pin'))] + '.xlsx'

pept_final.to_excel(xlsx_output_file, index=False)
pept_final.to_csv(csv_output_file, index=False)
pept_final.to_csv(output_file,sep='\t', index=False)

print("Completed.")