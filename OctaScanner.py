import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from Pfeature.pfeature import ddr_wp, dpc_wp
import os
import numpy as np
from stmol import showmol
import py3Dmol
import requests
from Bio import SeqIO
from io import StringIO
def main():
    # Set the color scheme
    header_color = '#91C788'
    background_color = '#FFFFFF'
    text_color = '#333333'
    primary_color = '#800000'
    footer_color = '#017C8C'
    footer_text_color = '#FFFFFF'
    font = 'Arial, sans serif'

    # Set the page config
    st.set_page_config(
        page_title='OctaScanner',
        layout='wide',
        initial_sidebar_state='expanded',
        page_icon='🎡',
    )

    # Set the theme
    st.markdown(f"""
    <style>
        .reportview-container {{
            background-color: {background_color};
            color: {text_color};
            font-family: {font};
        }}
        .sidebar .sidebar-content {{
            background-color: {header_color};
            color: {text_color};
        }}
        .stButton > button {{
            background-color: {primary_color};
            color: {background_color};
            border-radius: 12px;
            font-size: 16px;
            padding: 10px 20px;
        }}
        footer {{
            font-family: {font};
            background-color: {footer_color};
            color: {footer_text_color};
        }}
        .header-title {{
            color: {primary_color};
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }}
        .header-subtitle {{
            color: {text_color};
            font-size: 20px;
            text-align: center;
            margin-bottom: 30px;
        }}
    </style>
    """, unsafe_allow_html=True)
  

     # Add the image and title at the top of the page
    col1, col2, col3 = st.columns([1,2,3])
    with col1:
        st.image("hiv2.jpg", width=580)
    with col3:
        st.markdown("<h1 class='header-title'>OctaScanner – An Innovative Approach towards HIV Therapeutics</h1>", unsafe_allow_html=True)
        st.markdown("""
        <p class='header-subtitle'>
        Welcome to HIV OctaScanner, a prediction server designed to decipher the HIV Proteolytic activity.Following the unique blended approach of traditional machine learning with the depth of Deep Learning neural networks, 
    this algorithm unveils intricate molecular patterns previously unseen.Join us in the pursuit of scientific discovery to illuminate the mechanisms underlying HIV Protease functionality, 
    potentially unlocking new therapeutic avenues in the fight against this devastating virus.
        </p>
        """, unsafe_allow_html=True)
# Add university logos to the page
    left_logo, center, right_logo = st.columns([1, 2, 1])
    center.image("ref.jpg", width=650)
    #right_logo.image("image.jpg", width=250)

if __name__ == "__main__":
    main()

# Load the trained model
model_file = "rf_model.pkl"  # Ensure this path is correct
model = joblib.load(model_file)

if 'current_seq_idx' not in st.session_state:
    st.session_state.current_seq_idx = 0

def ddr(input_seq):
    input_file = 'input_seq.txt'
    output_file = 'output_ddr.csv'
    with open(input_file, 'w') as f:
        f.write(">input_sequence\n" + input_seq)
    ddr_wp(input_file, output_file)
    df = pd.read_csv(output_file)
    os.remove(input_file)
    os.remove(output_file)
    return df

def dpc(input_seq):
    input_file = 'input_seq.txt'
    output_file = 'output_dpc.csv'
    with open(input_file, 'w') as f:
        f.write(">input_sequence\n" + input_seq)
    dpc_wp(input_file, output_file, 1)
    df = pd.read_csv(output_file)
    os.remove(input_file)
    os.remove(output_file)
    return df

def is_valid_sequence(sequence):
    valid_amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    if not sequence or not all(char.upper() in valid_amino_acids for char in sequence):
        raise ValueError("You have entered an invalid sequence. Please check your input.")
    return True

def update(sequence_list):
    pdb_strings = []
    for sequence in sequence_list:
        # Convert the sequence to uppercase for API compatibility
        uppercase_sequence = sequence.upper()

        if not is_valid_sequence(uppercase_sequence):
            st.error(f"Invalid sequence: {sequence}")
            continue

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        response = requests.post('https://api.esmatlas.com/foldSequence/v1/pdb/', headers=headers, data=uppercase_sequence, verify=False)
        if response.status_code == 200:
            pdb_string = response.content.decode('utf-8')
            pdb_strings.append(pdb_string)
        else:
            st.error(f"Error with sequence {sequence}: Status code {response.status_code}")
    return pdb_strings


# Function to parse FASTA format
def parse_fasta(file_content):
    sequences = []
    current_sequence = ""
    for line in file_content:
        if line.startswith('>'):
            if current_sequence:
                sequences.append(current_sequence)
                current_sequence = ""
        else:
            current_sequence += line.strip()
    if current_sequence:
        sequences.append(current_sequence)
    return sequences


def predict_peptide_structure(sequences):
    ddr_df_list = [ddr(seq) for seq in sequences if seq]
    dpc_df_list = [dpc(seq) for seq in sequences if seq]
    df_features = pd.concat([pd.concat(ddr_df_list, axis=0),
                             pd.concat(dpc_df_list, axis=0)], axis=1)
    feature_cols = ['DDR_A', 'DDR_C', 'DDR_D', 'DDR_E', 'DDR_F', 'DDR_G', 'DDR_H', 'DDR_I', 'DDR_K', 'DDR_L', 'DDR_M', 'DDR_N', 'DDR_P', 'DDR_Q', 'DDR_R', 'DDR_S', 'DDR_T', 'DDR_V', 'DDR_W', 'DDR_Y', 'DPC1_AA', 'DPC1_AC', 'DPC1_AD', 'DPC1_AE', 'DPC1_AF', 'DPC1_AG', 'DPC1_AH', 'DPC1_AI', 'DPC1_AK', 'DPC1_AL', 'DPC1_AM', 'DPC1_AN', 'DPC1_AP', 'DPC1_AQ', 'DPC1_AR', 'DPC1_AS', 'DPC1_AT', 'DPC1_AV', 'DPC1_AW', 'DPC1_AY', 'DPC1_CA', 'DPC1_CC', 'DPC1_CD', 'DPC1_CE', 'DPC1_CF', 'DPC1_CG', 'DPC1_CH', 'DPC1_CI', 'DPC1_CK', 'DPC1_CL', 'DPC1_CM', 'DPC1_CN', 'DPC1_CP', 'DPC1_CQ', 'DPC1_CR', 'DPC1_CS', 'DPC1_CT', 'DPC1_CV', 'DPC1_CW', 'DPC1_CY', 'DPC1_DA', 'DPC1_DC', 'DPC1_DD', 'DPC1_DE', 'DPC1_DF', 'DPC1_DG', 'DPC1_DH', 'DPC1_DI', 'DPC1_DK', 'DPC1_DL', 'DPC1_DM', 'DPC1_DN', 'DPC1_DP', 'DPC1_DQ', 'DPC1_DR', 'DPC1_DS', 'DPC1_DT', 'DPC1_DV', 'DPC1_DW', 'DPC1_DY', 'DPC1_EA', 'DPC1_EC', 'DPC1_ED', 'DPC1_EE', 'DPC1_EF', 'DPC1_EG', 'DPC1_EH', 'DPC1_EI', 'DPC1_EK', 'DPC1_EL', 'DPC1_EM', 'DPC1_EN', 'DPC1_EP', 'DPC1_EQ', 'DPC1_ER', 'DPC1_ES', 'DPC1_ET', 'DPC1_EV', 'DPC1_EW', 'DPC1_EY', 'DPC1_FA', 'DPC1_FC', 'DPC1_FD', 'DPC1_FE', 'DPC1_FF', 'DPC1_FG', 'DPC1_FH', 'DPC1_FI', 'DPC1_FK', 'DPC1_FL', 'DPC1_FM', 'DPC1_FN', 'DPC1_FP', 'DPC1_FQ', 'DPC1_FR', 'DPC1_FS', 'DPC1_FT', 'DPC1_FV', 'DPC1_FW', 'DPC1_FY', 'DPC1_GA', 'DPC1_GC', 'DPC1_GD', 'DPC1_GE', 'DPC1_GF', 'DPC1_GG', 'DPC1_GH', 'DPC1_GI', 'DPC1_GK', 'DPC1_GL', 'DPC1_GM', 'DPC1_GN', 'DPC1_GP', 'DPC1_GQ', 'DPC1_GR', 'DPC1_GS', 'DPC1_GT', 'DPC1_GV', 'DPC1_GW', 'DPC1_GY', 'DPC1_HA', 'DPC1_HC', 'DPC1_HD', 'DPC1_HE', 'DPC1_HF', 'DPC1_HG', 'DPC1_HH', 'DPC1_HI', 'DPC1_HK', 'DPC1_HL', 'DPC1_HM', 'DPC1_HN', 'DPC1_HP', 'DPC1_HQ', 'DPC1_HR', 'DPC1_HS', 'DPC1_HT', 'DPC1_HV', 'DPC1_HW', 'DPC1_HY', 'DPC1_IA', 'DPC1_IC', 'DPC1_ID', 'DPC1_IE', 'DPC1_IF', 'DPC1_IG', 'DPC1_IH', 'DPC1_II', 'DPC1_IK', 'DPC1_IL', 'DPC1_IM', 'DPC1_IN', 'DPC1_IP', 'DPC1_IQ', 'DPC1_IR', 'DPC1_IS', 'DPC1_IT', 'DPC1_IV', 'DPC1_IW', 'DPC1_IY', 'DPC1_KA', 'DPC1_KC', 'DPC1_KD', 'DPC1_KE', 'DPC1_KF', 'DPC1_KG', 'DPC1_KH', 'DPC1_KI', 'DPC1_KK', 'DPC1_KL', 'DPC1_KM', 'DPC1_KN', 'DPC1_KP', 'DPC1_KQ', 'DPC1_KR', 'DPC1_KS', 'DPC1_KT', 'DPC1_KV', 'DPC1_KW', 'DPC1_KY', 'DPC1_LA', 'DPC1_LC', 'DPC1_LD', 'DPC1_LE', 'DPC1_LF', 'DPC1_LG', 'DPC1_LH', 'DPC1_LI', 'DPC1_LK', 'DPC1_LL', 'DPC1_LM', 'DPC1_LN', 'DPC1_LP', 'DPC1_LQ', 'DPC1_LR', 'DPC1_LS', 'DPC1_LT', 'DPC1_LV', 'DPC1_LW', 'DPC1_LY', 'DPC1_MA', 'DPC1_MC', 'DPC1_MD', 'DPC1_ME', 'DPC1_MF', 'DPC1_MG', 'DPC1_MH', 'DPC1_MI', 'DPC1_MK', 'DPC1_ML', 'DPC1_MM', 'DPC1_MN', 'DPC1_MP', 'DPC1_MQ', 'DPC1_MR', 'DPC1_MS', 'DPC1_MT', 'DPC1_MV', 'DPC1_MW', 'DPC1_MY', 'DPC1_NA', 'DPC1_NC', 'DPC1_ND', 'DPC1_NE', 'DPC1_NF', 'DPC1_NG', 'DPC1_NH', 'DPC1_NI', 'DPC1_NK', 'DPC1_NL', 'DPC1_NM', 'DPC1_NN', 'DPC1_NP', 'DPC1_NQ', 'DPC1_NR', 'DPC1_NS', 'DPC1_NT', 'DPC1_NV', 'DPC1_NW', 'DPC1_NY', 'DPC1_PA', 'DPC1_PC', 'DPC1_PD', 'DPC1_PE', 'DPC1_PF', 'DPC1_PG', 'DPC1_PH', 'DPC1_PI', 'DPC1_PK', 'DPC1_PL', 'DPC1_PM', 'DPC1_PN', 'DPC1_PP', 'DPC1_PQ', 'DPC1_PR', 'DPC1_PS', 'DPC1_PT', 'DPC1_PV', 'DPC1_PW', 'DPC1_PY', 'DPC1_QA', 'DPC1_QC', 'DPC1_QD', 'DPC1_QE', 'DPC1_QF', 'DPC1_QG', 'DPC1_QH', 'DPC1_QI', 'DPC1_QK', 'DPC1_QL', 'DPC1_QM', 'DPC1_QN', 'DPC1_QP', 'DPC1_QQ', 'DPC1_QR', 'DPC1_QS', 'DPC1_QT', 'DPC1_QV', 'DPC1_QW', 'DPC1_QY', 'DPC1_RA', 'DPC1_RC', 'DPC1_RD', 'DPC1_RE', 'DPC1_RF', 'DPC1_RG', 'DPC1_RH', 'DPC1_RI', 'DPC1_RK', 'DPC1_RL', 'DPC1_RM', 'DPC1_RN', 'DPC1_RP', 'DPC1_RQ','DPC1_RR', 'DPC1_RS', 'DPC1_RT', 'DPC1_RV', 'DPC1_RW', 'DPC1_RY', 'DPC1_SA', 'DPC1_SC', 'DPC1_SD', 'DPC1_SE', 'DPC1_SF', 'DPC1_SG', 'DPC1_SH', 'DPC1_SI', 'DPC1_SK', 'DPC1_SL', 'DPC1_SM', 'DPC1_SN', 'DPC1_SP', 'DPC1_SQ', 'DPC1_SR', 'DPC1_SS', 'DPC1_ST', 'DPC1_SV', 'DPC1_SW', 'DPC1_SY', 'DPC1_TA', 'DPC1_TC', 'DPC1_TD', 'DPC1_TE', 'DPC1_TF', 'DPC1_TG', 'DPC1_TH', 'DPC1_TI', 'DPC1_TK', 'DPC1_TL', 'DPC1_TM', 'DPC1_TN', 'DPC1_TP', 'DPC1_TQ', 'DPC1_TR', 'DPC1_TS', 'DPC1_TT', 'DPC1_TV', 'DPC1_TW', 'DPC1_TY', 'DPC1_VA', 'DPC1_VC', 'DPC1_VD', 'DPC1_VE', 'DPC1_VF', 'DPC1_VG', 'DPC1_VH', 'DPC1_VI', 'DPC1_VK', 'DPC1_VL', 'DPC1_VM', 'DPC1_VN', 'DPC1_VP', 'DPC1_VQ', 'DPC1_VR', 'DPC1_VS', 'DPC1_VT', 'DPC1_VV', 'DPC1_VW', 'DPC1_VY', 'DPC1_WA', 'DPC1_WC', 'DPC1_WD', 'DPC1_WE', 'DPC1_WF', 'DPC1_WG', 'DPC1_WH', 'DPC1_WI', 'DPC1_WK', 'DPC1_WL', 'DPC1_WM', 'DPC1_WN', 'DPC1_WP', 'DPC1_WQ', 'DPC1_WR', 'DPC1_WS', 'DPC1_WT', 'DPC1_WV', 'DPC1_WW', 'DPC1_WY', 'DPC1_YA', 'DPC1_YC', 'DPC1_YD', 'DPC1_YE', 'DPC1_YF', 'DPC1_YG', 'DPC1_YH', 'DPC1_YI', 'DPC1_YK', 'DPC1_YL', 'DPC1_YM', 'DPC1_YN', 'DPC1_YP', 'DPC1_YQ', 'DPC1_YR', 'DPC1_YS', 'DPC1_YT', 'DPC1_YV', 'DPC1_YW', 'DPC1_YY']
    df_features = df_features.reindex(columns=feature_cols, fill_value=0)
    y_pred = model.predict(df_features)
    prediction_probability = model.predict_proba(df_features)[:, 1]
    return y_pred, prediction_probability

# HTML and CSS to color the title
st.markdown(
    """
    <style>
    .title {
        color: #3BB143;  /* Parrot Green color code */
        font-size: 2em;
        font-weight: bold;
    }
    </style>
    <h1 class="title">Sequence Submission</h1>
    """,
    unsafe_allow_html=True
)


if 'page' not in st.session_state:
    st.session_state.page = 'input'
if 'submit_count' not in st.session_state:
    st.session_state.submit_count = 0

if st.session_state.page == 'input':
    st.subheader("Please Enter Octamers in FASTA Format")
    protein_sequences = st.text_area("Octamer Sequences (Enter multiple sequences separated by new lines)", height=150)
    fasta_file = st.file_uploader("Or upload FASTA file", type=["fasta", "txt"])

    submit_button = st.button("Submit", key="input_submit")

    if submit_button:
        st.session_state.submit_count += 1

    if fasta_file:
        fasta_content = fasta_file.getvalue().decode("utf-8").splitlines()
        protein_sequences = parse_fasta(fasta_content)
        st.info("File uploaded. Ready to submit.")
    else:
        protein_sequences = protein_sequences.strip().split('\n')

    if submit_button:
        if not protein_sequences:
            st.error("Please enter protein sequences or upload a FASTA file.")
        else:
            st.session_state.protein_sequences = protein_sequences
            predictions, prediction_probability = predict_peptide_structure(protein_sequences)
            st.session_state.prediction = predictions
            st.session_state.prediction_probability = prediction_probability
            st.session_state.page = 'output'

if st.session_state.page == 'output':
    st.subheader("Prediction Results")

    results_df = pd.DataFrame({
        'Index': range(1, len(st.session_state.protein_sequences) + 1),
        'Peptide Sequence': st.session_state.protein_sequences,
        'Predicted Probability': st.session_state.prediction_probability,
        'Class Label': st.session_state.prediction
    })

    st.table(results_df)

    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='prediction_results.csv',
        mime='text/csv',
    )

    st.button("Back", on_click=lambda: setattr(st.session_state, 'page', 'input'))
    structure_container = st.container()

# HTML and CSS to color the title and header
st.markdown(
    """
    <style>
    .title {
        color: #800000;  /* Parrot Green color code */
        font-size: 2em;
        font-weight: bold;
    }
    .header {
        color: #800000;  /* Parrot Green color code */
        font-size: 1.5em;
        font-weight: bold;
    }
    </style>
    <h1 class="title">Team OctaScanner:</h1>
    """,
    unsafe_allow_html=True
)


row1, row2 = st.columns([1, 1])
row3 = st.columns(1)

with row1:
    st.write("")
    st.write("### Dr. Kashif Iqbal Sahibzada")
    #st.write("Assistant Professor")
    st.write("Assistant Professor | Department of Health Professional Technologies, Faculty of Allied Health Sciences, The University of Lahore")
    st.write("Post-Doctoral Fellow | Henan University of Technology,Zhengzhou China ")
    st.write("Email: kashif.iqbal@dhpt.uol.edu.pk | kashif.iqbal@haut.edu.cn")
with row2:
 st.write("")
 st.write("### Rizwan Abid")
 st.write("PhD Scholar")
 st.write("School of Biochemistry and Biotechnology")
 st.write("University of the Punjab, Lahore")
 st.write("Email: rizwan.phd.ibb@pu.edu.pk")  


#Add University Logo
left_logo, center_left, center_right, right_logo = st.columns([1, 1, 1, 1])
#left_logo.image("LOGO_u.jpeg", width=200)
center_left.image("uol.jpg", width=450)  # Replace with your center-left logo image
#right_logo.image("image.jpg", width=200)

