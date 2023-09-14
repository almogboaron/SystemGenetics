import GEOparse
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm


def get_gse(gse_name: str) ->  GEOparse.GEOTypes.GSE:
    gse = GEOparse.get_GEO(geo=gse_name, annotate_gpl=True, include_data=True)
    return gse


def get_liver_gene_mapping(gse):
    gpl_name = list(gse.gpls.keys())[0]
    df = gse.gpls[gpl_name].table

    id_to_gene_assignment = {}

    for index, row in df.iterrows():
        ID = row['ID']
        gene_name = row['GENE_SYMBOL']
        if not gene_name or pd.isna(gene_name):
            continue
        id_to_gene_assignment[ID] = gene_name

    return id_to_gene_assignment


def get_hypo_gene_mapping(gse: GEOparse.GEOTypes.GSE) -> dict:
    gpl_name = list(gse.gpls.keys())[0]
    df = gse.gpls[gpl_name].table

    id_to_gene_assignment = {}

    for index, row in df.iterrows():
        ID = row['ID']
        gene_assignment = row["gene_assignment"]
        try:
            # Split the 'gene_assignment' value and use the second part as the value
            gene_name = gene_assignment.split(" // ")[1]
        except:
            # gene assignment is not valid - I saw "---"
            # print(gene_assignment)
            continue
        id_to_gene_assignment[ID] = gene_name

    return id_to_gene_assignment


def get_strain_name_from_gsm(gsm: GEOparse.GEOTypes.GSM, gse_type: str) -> str:
    if gse_type == "liver":
        strain_name = gsm.metadata["characteristics_ch1"][0].split(": ")[1]
    elif gse_type == "hypo":
        strain_name = gsm.metadata["characteristics_ch1"][1].split(": ")[1]
    else:
        raise Exception(f"Unsupported gse type: {gse_type} in get_strain_name_from_gsm")
    return strain_name


def parse_gse(gse: GEOparse.GEOTypes.GSE, gse_type: str) -> pd.DataFrame:
    """
    relevant only for liver and hypothalamus!!
    :param gse:
    :return:
    """
    if gse_type == "liver":
        id_ref_to_gene = get_liver_gene_mapping(gse)
    elif gse_type == "hypo":
        id_ref_to_gene = get_hypo_gene_mapping(gse)
    else:
        raise Exception(f"Unsupported gse type: {gse_type} in parse_gse")

    tables = []
    first = True
    for gsm_name, gsm in tqdm(gse.gsms.items()):
        strain_name = get_strain_name_from_gsm(gsm, gse_type)
        df = gsm.table.rename(columns={"VALUE": strain_name})
        valid_id_refs = df['ID_REF'].isin(id_ref_to_gene.keys())
        filtered_df = df[valid_id_refs]
        # filtered_df['ID_REF'] = filtered_df['ID_REF'].replace(id_ref_to_gene)
        # filtered_df.loc[:, 'ID_REF'] = filtered_df['ID_REF'].replace(id_ref_to_gene)
        if first:
            tables.append(filtered_df[["ID_REF", strain_name]])
            first = False
        else:
            tables.append(filtered_df[[strain_name]])

    full_df = pd.concat(tables, axis=1)
    full_df = full_df.dropna()
    full_df.loc[:, 'ID_REF'] = full_df['ID_REF'].replace(id_ref_to_gene)
    return full_df



def parse_soft_file(file_path: str) -> pd.DataFrame:
    with open(file_path, "r") as f:
        data = f.readlines()

    table_start_idx = table_end_idx = 0
    for i, l in enumerate(data):
        if l.startswith("!dataset_table_begin"):
            table_start_idx = i + 1
        elif l.startswith("!dataset_table_end"):
            table_end_idx = i

    table_rows = data[table_start_idx:table_end_idx]


def parse_tables():
    with open(r"data_sets\liver_data.pickle", 'rb') as f:
        liver_db = pickle.load(f)

    tables = []
    first = True
    for gsm_name, gsm in liver_db.gsms.items():
        df = gsm.table.rename(columns={"VALUE": gsm_name})
        df = df[df["Detection Pval"] <= 0.01]
        if first:
            tables.append(df[["ID_REF", gsm_name]])
            first = False
        else:
            tables.append(df[[gsm_name]])

    full_df = pd.concat(tables, axis=1)
    full_df = full_df.dropna()
    return full_df


def generate_working_dfs():
    liver_gse_name = "GSE17522"
    liver_gse = get_gse(liver_gse_name)
    liver_df = parse_gse(liver_gse, "liver")
    liver_df.to_csv("liver_processed_data.csv", index=False)

    hypo_gse_name = "GSE36674"
    hypo_gse = get_gse(hypo_gse_name)
    hypo_df = parse_gse(hypo_gse, "hypo")
    hypo_df.to_csv("hypo_processed_data.csv", index=False)


def take_avg_on_multi_column_breeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gets df with multiple same BXD columns, like BXD60, BXD60.1
    And makes one column for breed with average value
    :param df:
    :return:
    """
    just_breeds = df.drop(columns=["ID_REF"])
    averages = just_breeds.groupby(lambda col: col.split('.')[0], axis=1).mean()
    averages.columns = [col.split('.')[0] if '.' in col else col for col in averages.columns]
    result_df = pd.concat([df['ID_REF'], averages], axis=1)
    return result_df


def filter_neighboring_rows(data_frame, columns_to_check):
    # Create a boolean mask to identify rows where any of the specified columns differ from the previous row
    mask = ~data_frame[columns_to_check].eq(data_frame[columns_to_check].shift(1)).all(axis=1)
    filtered_df = data_frame[mask]
    return filtered_df


def process_data(csv_file: str) -> pd.DataFrame:
    """
    Get a processed data csv with genes and strains (BXDs)
    And clean and prep it for further analysis
    :param csv_file:
    :return:
    """
    df = pd.read_csv(csv_file)

    # remove index column incase it was saved in the csv
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    # remove rows with no gene identifier
    df.dropna(subset=['ID_REF'], inplace=True)

    # make df with unique breeds only (take avg of same BXDs)
    df = take_avg_on_multi_column_breeds(df)

    # Remove rows with low maximal value
    max_values = df.iloc[:, 1:].max(axis=1)  # Assuming expression columns start from the second column
    percentile_threshold = 90  # means we take the upper 10% of values
    value_threshold = np.percentile(max_values, percentile_threshold)
    filtered_df = df[max_values >= value_threshold]

    # Remove rows with low variance
    variance_values = filtered_df.iloc[:, 1:].var(axis=1)
    percentile_threshold = 90
    value_threshold = np.percentile(variance_values, percentile_threshold)
    filtered_df = filtered_df[variance_values >= value_threshold]

    # taking one probe with highest variance
    filtered_df['RowVar'] = filtered_df.iloc[:, 1:].var(axis=1)
    sorted_df = filtered_df.sort_values(by=['ID_REF', 'RowVar'], ascending=[True, False])
    result_df = sorted_df.drop_duplicates(subset='ID_REF', keep='first').reset_index(drop=True)
    result_df = result_df.drop(columns=['RowVar'])

    # filter neighboring loci
    res_df = filter_neighboring_rows(result_df, result_df.columns.delete([0]))

    return res_df


def checks():
    hypo_ready_df = process_data("hypo_processed_data.csv")
    hypo_ready_df.to_csv("hypo_ready.csv", index=False)

    liver_ready_df = process_data("liver_processed_data.csv")
    liver_ready_df.to_csv("liver_ready.csv", index=False)



def correct_parsing():
    with open(r"data_sets\liver_data.pickle", 'rb') as f:
        gse = pickle.load(f)

    gpl_data = gse.gpls[list(gse.gpls.keys())[0]]
    df = gpl_data.table


def get_expression_data():
    hsc_db = GEOparse.get_GEO(filepath=r"data_sets\GDS1077_full.soft")
    hsc_df = hsc_db.table
    hsc_df.to_csv("hsc_df.csv")

    with open(r"data_sets\liver_data.pickle", 'rb') as f:
        gse = pickle.load(f)

    gpl_data = gse.gpls[list(gse.gpls.keys())[0]]
    liver_df = gpl_data.table
    liver_df.to_csv("liver_df.csv")


def test_GEOparse():
    # Load the SOFT file for the specified dataset ID (GDS number)
    # db1 = GEOparse.get_GEO(filepath=r"data_sets\GDS1077_full.soft")
    # db2 = GEOparse.get_GEO(filepath=r"data_sets\GSE18067_family.soft")

    # write pickle for later use
    # with open(r"data_sets\liver_data.pickle", 'wb') as f:
    #     pickle.dump(db2, f)

    hsc_dataset_id = "GDS1077"
    hsc_dataset = GEOparse.get_GEO(hsc_dataset_id)

    liver_dataset_id = "GSE17522"
    liver_dataset = GEOparse.get_GEO(liver_dataset_id)

    # Access the expression data and metadata
    expression_data = dataset.pivot_table
    metadata = dataset.metadata

    # Filter the expression data to retain only BXD samples
    bxd_sample_ids = [sample_id for sample_id in expression_data.columns if sample_id.startswith("BXD")]
    bxd_expression_data = expression_data[bxd_sample_ids]

    # Display the extracted BXD expression data
    print(bxd_expression_data)


if __name__ == '__main__':
    # test_GEOparse()
    # parse_tables()
    # correct_parsing()
    # generate_working_dfs()
    checks()
    pass