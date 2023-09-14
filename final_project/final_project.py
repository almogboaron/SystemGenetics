import GEOparse
import pandas as pd
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
    liver_df.to_csv("liver_processed_data.csv")

    hypo_gse_name = "GSE36674"
    hypo_gse = get_gse(hypo_gse_name)
    hypo_df = parse_gse(hypo_gse, "hypo")
    hypo_df.to_csv("hypo_processed_data.csv")


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
    generate_working_dfs()
    pass