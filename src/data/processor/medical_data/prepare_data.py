import argparse
import os
import pandas as pd
from os.path import join, dirname, abspath

project_root_path = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))
structured_data_path = join(project_root_path, 'data', 'structured')
processed_dir = join(structured_data_path, 'prad', 'processed')
data_dir = join(structured_data_path, 'prad', 'raw_data')


def configure_cohort(cancer_type):
    """Select the cohort-specific raw and processed directories."""
    global processed_dir, data_dir
    cohort = {'prostate': 'prad', 'breast': 'brca'}[cancer_type]
    cohort_dir = join(structured_data_path, cohort)
    processed_dir = join(cohort_dir, 'processed')
    data_dir = join(cohort_dir, 'raw_data')
    os.makedirs(processed_dir, exist_ok=True)
    return cohort_dir




def prepare_design_matrix_crosstable(study_names):
    print('preparing mutations table...')

    df_tables = None
    for study in study_names:
        filename = 'data_mutations.txt'
        id_col = 'Tumor_Sample_Barcode'
        study_dir = join(data_dir, study)
        df = pd.read_csv(join(study_dir, filename), sep='\t', low_memory=False)
        
        mutation_dis = df['Variant_Classification'].value_counts()
        print(f'{study} mutation distribution is \n {mutation_dis}')

        if filter_silent_muts:
            df = df[df['Variant_Classification'] != 'Silent'].copy()
        if filter_missense_muts:
            df = df[df['Variant_Classification'] != 'Missense_Mutation'].copy()
        if filter_introns_muts:
            df = df[df['Variant_Classification'] != 'Intron'].copy()

        # important_only = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Splice_Site','Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Start_Codon_SNP','Nonstop_Mutation', 'De_novo_Start_OutOfFrame', 'De_novo_Start_InFrame']
        exclude = ['Silent', 'Intron', "3\'UTR", "5\'UTR", 'RNA', 'lincRNA']
        if keep_important_only:
            df = df[~df['Variant_Classification'].isin(exclude)].copy()
        if truncating_only:
            include = ['Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins']
            df = df[df['Variant_Classification'].isin(include)].copy()
        df_table = pd.pivot_table(data=df, index=id_col, columns='Hugo_Symbol', values='Variant_Classification',
                                aggfunc='count')
        df_table = df_table.fillna(0)
        total_numb_mutations = df_table.sum().sum()

        number_samples = df_table.shape[0]
        print('number of mutations', total_numb_mutations, total_numb_mutations / (number_samples + 0.0))
        if df_tables is None:
            df_tables = df_table
        else:
            if intersection == True:
                common_cols = list(set(df_table.columns) & set(df_tables.columns))
                df_table, df_tables = df_table[common_cols], df_tables[common_cols]
                df_tables = pd.concat((df_tables, df_table), axis=0)  # keep columns that both contain.
            else:
                df_tables = pd.concat((df_tables, df_table), axis=0, join='outer')  # keep all columns, fill 0 in nan position.
                df_tables.fillna(0)
    df_tables = df_tables.loc[~df_tables.index.duplicated(keep='first')]
    filename = join(processed_dir, 'final_analysis_set_cross_' + ext + '.csv')
    df_tables.to_csv(filename)


def prepare_response(study_names):
    print('preparing response table...')
    df_tables = None
    for study in study_names:
        filename = 'data_clinical_sample.txt'
        study_dir = join(data_dir, study)
        df = pd.read_csv(join(study_dir, filename), sep='\t', low_memory=False, skiprows=4)
        response = pd.DataFrame()
        response['id'] = df['SAMPLE_ID']
        if study == 'brca_mbcproject_wagle_2017':
            response['response'] = df['CALC_MET_SETTING']
            response = response[
                (response['response'] == 'METASTATIC DISEASE PRESENT') | 
                (response['response'] == 'NO METASTATIC DISEASE PRESENT')
            ]
            response['response'] = response['response'].replace('METASTATIC DISEASE PRESENT', 1)
            response['response'] = response['response'].replace('NO METASTATIC DISEASE PRESENT', 0)
        elif study == 'brca_mbcproject_2022':
            response['response'] = df['CALC_MET_SETTING']
            response = response[
                (response['response'] == 'METASTATIC_DISEASE_PRESENT') | 
                (response['response'] == 'NO_METASTATIC_DISEASE_PRESENT')
            ]
            response['response'] = response['response'].replace('METASTATIC_DISEASE_PRESENT', 1)
            response['response'] = response['response'].replace('NO_METASTATIC_DISEASE_PRESENT', 0)
        else:
            sample_type = df['SAMPLE_TYPE'].astype(str).str.strip().str.upper()
            response['response'] = sample_type.map({
                'METASTASIS': 1,
                'METASTATIC': 1,
                'METASTATIC TUMOR': 1,
                'PRIMARY': 0,
                'PRIMARY TUMOR': 0,
            })
            response = response[response['response'].notna()].copy()
        response = response.drop_duplicates()
        response = response.dropna(axis=0, how='all')
        if df_tables is None:
            df_tables = response
        else:
            df_tables = pd.concat((df_tables, response), axis=0)
    df_tables = df_tables.loc[~df_tables['id'].duplicated(keep='first')]
    unexpected = set(df_tables['response'].dropna().unique()) - {0, 1}
    if unexpected:
        raise ValueError(f'Unexpected response labels after normalization: {unexpected}')
    df_tables['response'] = df_tables['response'].astype(int)
    df_tables.to_csv(join(processed_dir, 'response_paper.csv'), index=False)


def prepare_cnv(study_names):
    print('preparing copy number variants ...')
    df_tables = None
    for study in study_names:
        filename = 'data_cna.txt'
        study_dir = join(data_dir, study)
        df = pd.read_csv(join(study_dir, filename), sep='\t', low_memory=False, index_col=0)
        if study == 'brca_tcga':
            df = df.drop('Entrez_Gene_Id', axis=1)
        elif study == 'brca_igr_2015':
            df = df.drop('Entrez_Gene_Id', axis=1)
        elif study == 'brca_tcga_pan_can_atlas_2018':
            df = df.drop('Entrez_Gene_Id', axis=1)
        elif study == 'brca_metabric':
            df = df.drop('Entrez_Gene_Id', axis=1)
        elif study == 'brca_tcga_pub2015':
            df = df.drop('Entrez_Gene_Id', axis=1)
        elif study == 'brca_tcga_pub':
            df = df.drop('Entrez_Gene_Id', axis=1)
        elif study == 'prad_p1000':
            df = df.drop('Entrez_Gene_Id', axis=1)
        elif study == 'brca_mbcproject_wagle_2017':
            df = df.drop('Entrez_Gene_Id', axis=1)
        elif study == 'brca_mbcproject_2022':
            df = df.drop('Cytoband', axis=1)
        df = df.T
        df = df.fillna(0.)
        df.index.name = 'sample_id'
        # tmp = df.loc[:,df.columns.duplicated(keep=False)]
        # tmp.to_csv(join(processed_dir, 'duplicated.csv'))
        df = df.loc[~df.index.duplicated(keep='first')]
        df = df.loc[:,~df.columns.duplicated(keep='first')]

        print(f"{study} copy number variation distribution")
        flatten =  df.stack()
        print(flatten.value_counts())
        
        if df_tables is None:
            df_tables = df
        else:
            if intersection == True:
                # 合并两张表df, df_tables，按行合并，保留两张表共同的列
                df_tables = pd.concat([df_tables, df], axis=0, join='inner')
                # common_cols = list(set(df.columns) & set(df_tables.columns))
                # df, df_tables = df[common_cols], df_tables[common_cols]
                # df_tables = pd.concat([df_tables, df], axis=0)
            else:
                df_tables = pd.concat((df_tables, df), axis=0, join='outer')
                df_tables.fillna(0)
    df_tables = df_tables.loc[~df_tables.index.duplicated(keep='first')]
    filename = join(processed_dir, 'data_CNA_paper.csv')
    df_tables.to_csv(filename)


# def prepare_cnv_burden():
#     print('preparing copy number burden ...')
#     filename = '41588_2018_78_MOESM5_ESM.xlsx'
#     df = pd.read_excel(join(data_dir, filename), skiprows=2, index_col=1)
#     cnv = df['Fraction of genome altered']
#     filename = join(processed_dir, 'P1000_data_CNA_burden.csv')
#     cnv.to_frame().to_csv(filename)


# remove silent and intron mutations
filter_silent_muts = False
filter_missense_muts = False
filter_introns_muts = False
keep_important_only = True
truncating_only = False
intersection = True

ext = ""
if keep_important_only:
    ext = 'important_only'

if truncating_only:
    ext = 'truncating_only'

if filter_silent_muts:
    ext = "_no_silent"

if filter_missense_muts:
    ext = ext + "_no_missense"

if filter_introns_muts:
    ext = ext + "_no_introns"

def analyzing_data(study_name):
    data_mutation = pd.read_csv(join(processed_dir, 'final_analysis_set_cross_important_only.csv'), index_col=0, low_memory=False)
    data_cnv = pd.read_csv(join(processed_dir, 'data_CNA_paper.csv'), index_col=0, low_memory=False)
    data_response = pd.read_csv(join(processed_dir, 'response_paper.csv'), index_col=0)
    df = pd.read_csv(
        join(
            join(data_dir, study_name), "data_mutations.txt"
        ), 
        sep='\t', 
        low_memory=False
    )

    statistic_path = join(processed_dir, "statistic.csv")
    if os.path.exists(statistic_path):
        statistic = pd.read_csv(statistic_path, index_col=0)
    else:
        statistic = pd.DataFrame()

    if study_name not in statistic.index:
        statistic.loc[study_name] = pd.Series(dtype='object')

        statistic.loc[study_name, 'number_sample'] = data_response.shape[0]
        statistic.loc[study_name, 'sample_distribution'] = data_response.value_counts().to_string()
        statistic.loc[study_name, 'number_gene_tumation'] = data_mutation.shape[1]
        statistic.loc[study_name, 'number_gene_cnv'] = data_cnv.shape[1]
        statistic.loc[study_name, 'tumation_distribution'] = df['Variant_Classification'].value_counts().to_string()
        statistic.loc[study_name, 'total_number_mutations'] = data_mutation.sum().sum()
        statistic.loc[study_name, 'average_mutaions_per_sample'] = \
            statistic.loc[study_name, 'total_number_mutations'] / data_mutation.shape[0]
        statistic.loc[study_name, 'average_mutaions_per_gene'] = \
            statistic.loc[study_name, 'total_number_mutations'] / data_mutation.shape[1]
        statistic.loc[study_name, 'cnv_distribution'] = data_cnv.stack().value_counts().to_string()

        print(f"statictical results for {study_name}\n", statistic.loc[study_name])
        statistic.to_csv(statistic_path)

def prepare_data(study_names, cancer_type=None):
    if cancer_type is not None:
        configure_cohort(cancer_type)
    
    # study_names = ['breast_msk_2018']
    # study_names = ['prad_p1000']
    prepare_response(study_names)
    prepare_design_matrix_crosstable(study_names)
    prepare_cnv(study_names)
    # prepare_cnv_burden()
    if isinstance(study_names, list) and len(study_names) > 1:
        print(
            "Combined cohort matrices were prepared. Per-study descriptive "
            "statistics are skipped for multi-study cohorts."
        )
    elif isinstance(study_names, list) and len(study_names) == 1:
        analyzing_data(study_names[0])
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare cBioPortal matrices for KEPPNet.')
    parser.add_argument('--cancer_type', choices=['prostate', 'breast'], required=True)
    parser.add_argument('--study_names', nargs='+', required=True)
    args = parser.parse_args()
    prepare_data(args.study_names, cancer_type=args.cancer_type)
