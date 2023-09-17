from ex2.hw2_208273672 import q_2_analysis, plot_q2_results, generate_qtl_dict
from ex3.ex3 import association_test, analyse_eQTL_dict, filter_weak_associated_genes
import GEOparse
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from math import log10
from math import fabs


P_VALUE_THREASHOLD = 7.273


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
    # Todo: view distibution of max values to help choose better threshold
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

    # filter neighboring loci happens for genotypes xls
    # result_df = filter_neighboring_rows(result_df, result_df.columns.delete([0]))

    # change ID_REF column name to data
    result_df = result_df.rename(columns={'ID_REF': 'data'})

    return result_df


def pre_process_raw_dfs():
    hypo_ready_df = process_data("hypo_processed_data.csv")
    hypo_ready_df.to_csv("hypo_ready.csv", index=False)

    liver_ready_df = process_data("liver_processed_data.csv")
    liver_ready_df.to_csv("liver_ready.csv", index=False)


def eqtl_generation():
    liver_df = pd.read_csv("liver_ready.csv")
    association_test(expression_df=liver_df, dropped_file_name="liver_eqtl_dict.pickle")

    hypo_df = pd.read_csv("hypo_ready.csv")
    association_test(expression_df=hypo_df, dropped_file_name="hypo_eqtl_dict.pickle")


def filter_weak_associated_genes(dict_eQTL: dict, p_value_threshold: float) -> dict:
    res = {}
    for gene in dict_eQTL.keys():
        df_gene = pd.DataFrame(dict_eQTL[gene].items(), columns=['Locus', 'P_value'])
        df_filtered = df_gene[df_gene['P_value'] > p_value_threshold]
        if len(df_filtered) > 0:
            res[gene] = df_filtered
    return res


def analyze_genes_snps(genes_to_snps: dict):
    snp_count_dict = {}

    # Iterate over the gene_dict
    for gene, df in genes_to_snps.items():
        # Extract unique SNPs from each gene's DataFrame
        unique_snps = df['Locus'].unique()

        # Update the snp_count_dict with the count of unique SNPs
        for snp in unique_snps:
            snp_count_dict[snp] = snp_count_dict.get(snp, 0) + 1

    # Convert the snp_count_dict to a DataFrame (optional)
    result_df = pd.DataFrame(list(snp_count_dict.items()), columns=['SNP', 'Gene_Count'])
    return result_df


def eqtl_analysis(tissue="liver", file_path=None):
    if file_path:
        print(f"analyzing {file_path}")
        with open(file_path, 'rb') as f:
            eqtl_dict = pickle.load(f)

    else:
        print(f"analyzing {tissue} eQTLS")
        with open(f"{tissue}_eqtl_dict.pickle", 'rb') as f:
            eqtl_dict = pickle.load(f)

    # filter SNPs from each gene
    genes_relevant_snps = filter_weak_associated_genes(eqtl_dict, -log10(0.05))
    print(f"Number of unique genes / phenotypes with significant SNPs: {len(genes_relevant_snps)}")
    if file_path:
        print(f"Phenotypes with significant SNPs: {list(genes_relevant_snps.keys())}")
    snp_counts = [len(df) for k, df in genes_relevant_snps.items()]
    print(f"Avg number of significant SNPs per gene / phenotype: {np.average(snp_counts)}")
    print(f"Max number of significant SNPs per gene / phenotype: {np.max(snp_counts)}")
    print(f"Min number of significant SNPs per gene / phenotype: {np.min(snp_counts)}")

    res = analyze_genes_snps(genes_relevant_snps)
    print(f"Number of significant SNPs: {len(res)}")
    print(f"Maximum genes number associated by one SNP: {res['Gene_Count'].max()}")
    print(f"Minimum genes number associated by one SNP: {res['Gene_Count'].min()}")


def qtl_generation():
    # old phenotypes attempt
    # phenotypes_ids = [147, 114, 225, 231, 640, 2365, 2258, 685]

    # new chosen phenotypes
    phenotypes_ids = [260, 148, 2, 355, 356, 360, 1703, 684, 701, 703, 1719, 1822, 1880, 1885, 1953, 1954, 2010, 2034, 2156, 2177, 2179, 2195, 114, 142, 159, 160]
    generate_qtl_dict(phenotypes_ids, "genotypes.xls", "phenotypes.xls")


def revert_dict_get_snp_keys(genes_to_snps: dict) -> dict:
    snp_to_genes = {}
    for gene, df in genes_to_snps.items():
        unique_snps = df['Locus'].unique()
        for snp in unique_snps:
            existing_map = snp_to_genes[snp] = snp_to_genes.get(snp, [])
            existing_map.append(gene)
    return snp_to_genes


def compare_qtl_vs_eqtl(gene_to_snp: dict, phenotype_to_snp: dict):
    snp_to_genes = revert_dict_get_snp_keys(gene_to_snp)
    snp_to_phenotype = revert_dict_get_snp_keys(phenotype_to_snp)

    snps_affect_both = {}
    snps_only_affect_gene_expression = {}
    for snp, genes_ls in snp_to_genes.items():
        if snp in snp_to_phenotype:
            # print(f"Found SNP: {snp} that affects gene expression and phenotype")
            snps_affect_both[snp] = [genes_ls, snp_to_phenotype[snp]]
        else:
            snps_only_affect_gene_expression[snp] = genes_ls

    snps_affect_only_phenotype = {}
    for snp, phenotype in snp_to_phenotype.items():
        if snp not in snp_to_genes:
            snps_affect_only_phenotype[snp] = phenotype

    print(f"Number of SNPs affect both gene expression and phenotye: {len(snps_affect_both)}")
    print(f"SNPs affect both: {snps_affect_both}")
    print()
    print(f"Number of SNPs affect only gene expression: {len(snps_only_affect_gene_expression)}")
    print(f"Number of SNPs affect only phenotype: {len(snps_affect_only_phenotype)}")
    return snp_to_genes, snp_to_phenotype


def combine_results():
    with open("liver_eqtl_dict.pickle", 'rb') as f:
        liver_eqtl_dict = pickle.load(f)

    with open("hypo_eqtl_dict.pickle", 'rb') as f:
        hypo_eqtl_dict = pickle.load(f)

    with open("phenotypes_qtl_dict.pickle", 'rb') as f:
        phenotypes_qtl_dict = pickle.load(f)

    phenotypes_qtls = filter_weak_associated_genes(phenotypes_qtl_dict, -log10(0.05))

    liver_eqtls = filter_weak_associated_genes(liver_eqtl_dict, -log10(0.05))
    hypo_eqtls = filter_weak_associated_genes(hypo_eqtl_dict, -log10(0.05))


    print("Comparing eQTLs and QTLs for liver")
    snp_to_genes, snp_to_phenotypes = compare_qtl_vs_eqtl(liver_eqtls, phenotypes_qtls)

    print("Comparing eQTLs and QTLs for hypo")
    snp_to_genes, snp_to_phenotypes = compare_qtl_vs_eqtl(hypo_eqtls, phenotypes_qtls)


def Create_Triplets(snp_pheno_dict:dict ,snp_gene_dict:dict):
    geno_df = pd.read_excel(r"genotypes.xls")
    geno_df.set_index('Locus', inplace=True)
    set_triplets = set()
    for snp_pheno in snp_pheno_dict.keys():
        for pheno in snp_pheno_dict[snp_pheno]:
            for snp_gene in snp_gene_dict.keys():
                if((geno_df.loc[snp_pheno,"Chr_Build37"] == geno_df.loc[snp_gene,"Chr_Build37"]) and
                        fabs((geno_df["Build37_position"].loc[snp_pheno], - geno_df.loc["Build37_position"].loc[snp_gene])) < 2*10**6):
                            for gene in snp_gene_dict[snp_gene]:
                                set_triplets.add(np.array([snp_pheno, gene, pheno]))

    return set_triplets


def Df_For_Triplet(triplet:np.array,database:str)->pd.DataFrame:
    if(database=="hypo"):
        expression_df = pd.read_csv(r"hypo_ready.csv")
    if(database=="liver"):
        expression_df = pd.read_csv(r"liver_ready.csv")

    genotype_df = pd.read_excel(r"genotypes.xls", header=1)
    phenotype_df = pd.read_excel(r"phenotypes.xls")

    # Get genotype_row with data
    df_res = genotype_df.loc[genotype_df['Locus'] == triplet[0]]
    df_res = df_res.drop(df_res.columns[:4], axis=1)

    # Find columns with values other than 'B' or 'D'
    cols_to_drop = df_res.columns[~df_res.isin(['B', 'D']).all()]

    # Drop the identified columns
    df_res = df_res.drop(cols_to_drop, axis=1)


    df_res.replace({'B': 0, 'D': 1}, inplace=True)
    df_res.index = [r"B=0\D=1"]

    # Get expression row with data
    expression_row = expression_df.loc[expression_df['data'] == triplet[1]]
    expression_row = expression_row.drop(expression_row.columns[:2], axis=1)
    expression_row.index = ["R"]

    # Intesect columns and add to df_res:
    common_columns = df_res.columns.intersection(expression_row.columns)
    df_res = pd.concat([df_res[common_columns],expression_row[common_columns]],ignore_index=True)

    # Get Phenotype row with data
    phenotype_row = phenotype_df.loc[phenotype_df['ID_FOR_CHECK'] == int(triplet[2])]
    phenotype_row = phenotype_row.drop(phenotype_row.columns[:8], axis=1)
    phenotype_row.index = ["C"]

    # Intesect columns and add to df_res:
    common_columns = df_res.columns.intersection(phenotype_row.columns)
    df_res = pd.concat([df_res[common_columns],phenotype_row[common_columns]],ignore_index=True)


def permutation_test(base_df: pd.DataFrame, num_permutations: int = 100):
    res_dict = {}
    print("calculating probabilities for permutations")
    for i in tqdm(range(num_permutations)):
        shuffled_R = np.random.permutation(base_df['R'])
        shuffled_C = np.random.permutation(base_df['C'])

        shuffled_df = base_df.copy()
        shuffled_df['R'] = shuffled_R
        shuffled_df['C'] = shuffled_C

        probability = calculate_probabilities_for_df(shuffled_df)
        res_dict[i] = probability

    res_df = pd.DataFrame({"permutation": res_dict.keys(), "probability": res_dict.values()})
    return res_df


def check_triplet_significance(triplet: np.array, lr: float, permutations_lr: pd.DataFrame):
    mean_permutation_LR = np.mean(permutations_lr)  # Mean of permutation LR_ratios
    std_permutation_LR = np.std(permutations_lr)  # Standard deviation of permutation LR_ratios
    z_score = (lr - mean_permutation_LR) / std_permutation_LR

    # Calculate the p-value using the CDF of the standard normal distribution
    p_value = 1 - norm.cdf(z_score)



def analyze_causality():
    with open("snp_pheno_dict.pickle", 'rb') as f:
        snp_pheno_dict = pickle.load(f)

    with open("snp_liver_ge_dict.pickle", 'rb') as f:
        snp_liver_ge_dict = pickle.load(f)

    with open("snp_hypo_ge_dict.pickle", 'rb') as f:
        snp_hypo_ge_dict = pickle.load(f)

    triplets_set = Create_Triplets(snp_pheno_dict, snp_liver_ge_dict)
    for t in triplets_set:
        df = Df_For_Triplet(t, "liver")
        data_lr = calculate_probabilities_for_df(df)
        permutations_lr = permutation_test(df, num_permutations=100)





if __name__ == '__main__':
    # test_GEOparse()
    # parse_tables()
    # correct_parsing()
    # generate_working_dfs()
    # pre_process_raw_dfs()
    # eqtl_generation()
    # eqtl_analysis(tissue="liver")
    # eqtl_analysis(tissue="hypo")
    eqtl_analysis(file_path="phenotypes_qtl_dict.pickle")
    # qtl_generation()
    # combine_results()
    pass