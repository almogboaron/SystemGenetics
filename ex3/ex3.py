import pandas as pd
from ex2.hw2_208273672 import regression_model, prep_data, prep_genotype_data
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns


def add_together_same_bxd(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate the averages for columns with the same name but different indices
    just_breeds = df.drop(columns=["data"])
    averages = just_breeds.groupby(lambda col: col.split('_')[0], axis=1).mean()

    # Rename the columns for clarity
    averages.columns = [col.split('_')[0] if '_' in col else col for col in averages.columns]

    # Concatenate 'data' column from the original DataFrame with the averages
    result_df = pd.concat([df['data'], averages], axis=1)

    return result_df


def filter_neighboring_rows(data_frame, columns_to_check):
    # Create a boolean mask to identify rows where any of the specified columns differ from the previous row
    mask = ~data_frame[columns_to_check].eq(data_frame[columns_to_check].shift(1)).all(axis=1)

    # Apply the mask to filter the DataFrame
    filtered_df = data_frame[mask]

    return filtered_df


# PLOT_FOLDER = r"C:\Users\User1\Desktop\stuff\TAU\TASHPC\sem_b\systems_genetics\ex3\plots"   # Todo: change to generic absolute path
PLOT_FOLDER = r"C:\Users\User1\Desktop\stuff\TAU\TASHPC\sem_b\systems_genetics\git_stuff\SystemGenetics\final_project\plots"   # Todo: change to generic absolute path


def save_plot(df: pd.DataFrame, gene_name):
    p_val_c_name = "-log(p-value)"
    snp_c_name = "snp"

    # Sort the DataFrame by -log(p-value) in descending order
    # df.sort_values(by=p_val_c_name, ascending=False, inplace=True)

    # Plot the Manhattan plot
    # plt.figure(figsize=(10, 6))
    plt.scatter(range(len(df)), df[p_val_c_name])
    plt.axhline(df[p_val_c_name].mean(), color='red', linestyle='--')  # Add a red line for mean log p-value
    plt.xlabel('SNP Position')
    plt.ylabel('-log P-value')
    plt.title('Manhattan Plot')
    plt.xticks(range(len(df)), df[snp_c_name], rotation=90)  # Show SNP names on x-axis

    plt.savefig(f"{os.path.join(PLOT_FOLDER, gene_name + '.png')}", bbox_inches='tight')  # 'bbox_inches' helps prevent cropping of labels
    plt.close()  # Close the plot to free up memory


def association_test(expression_df: pd.DataFrame = None, dropped_file_name: str = "eqtl_res_dict.pickle"):
    """
    performs assocaition test between genotypes and expression data
    :return:
    """
    # load genotypes
    gen_df = pd.read_excel("genotypes.xls", header=1)

    # we want to filter the neighbour locis with the same genotype
    columns = gen_df.columns.delete([0, 1, 2, 3])
    gen_df = filter_neighboring_rows(gen_df, columns)

    # load expresion data
    if expression_df is None:
        lps_df = pd.read_csv("dendritic_LPS_stimulation.txt", sep="\t")
        exp_data = add_together_same_bxd(lps_df)
    else:
        exp_data = expression_df
    # exp_data.drop(columns=["B6", "D2"], inplace=True)
    relevant_breeds_names = set(exp_data.drop(columns=["data"]).columns)

    # generate dict of snp to breeds and allels
    genotypes_to_consider = ['B', 'D', 'H']
    snp_to_gn = {}
    for i, row in gen_df.iterrows():
        snp = row["Locus"]
        gn_filtered_breeds = {}
        for br in relevant_breeds_names:
            if br in gen_df.columns and row[br] in genotypes_to_consider:
                gn_filtered_breeds[br] = row[br]
        snp_to_gn[snp] = gn_filtered_breeds

    # generate the e_qtl dict
    e_qtl_dict = {}
    for index, row in exp_data.iterrows():
        data_value = row['data']
        filtered_row_dict = {col: row[col] for col in relevant_breeds_names}
        e_qtl_dict[data_value] = filtered_row_dict

    # Now for each snp we have a dict with breeds and their Allele

    eqtl_res = {}
    for eQTL in tqdm(e_qtl_dict):
        snp_to_res = {}
        for snp in tqdm(snp_to_gn):
            # run regression on each SNP and save the result which is -log(p-value)
            expression_data = e_qtl_dict[eQTL]
            s1 = set(snp_to_gn[snp].keys())
            s2 = set(expression_data.keys())
            not_in_concat = s1 ^ s2
            for k in not_in_concat:
                if k in expression_data:
                    expression_data.pop(k)
                elif k in snp_to_gn[snp]:
                    snp_to_gn[snp].pop(k)

            df = prep_data(snp_to_gn[snp], expression_data)
            prep_genotype_data(df, genotypes_to_consider)
            res = regression_model(df["genotype"], df["phenotype"])
            snp_to_res[snp] = res   # res is -log(p-value)

        eqtl_res[eQTL] = snp_to_res
        res_df = pd.DataFrame({"snp": eqtl_res[eQTL].keys(), "-log(p-value)": eqtl_res[eQTL].values()})
        save_plot(res_df, eQTL)

    with open(dropped_file_name, 'wb') as f:
        pickle.dump(eqtl_res, f)

    print("finish")


def filter_weak_associated_genes(dict_eQTL: dict, p_value_threshold: float) -> dict:
    res = {}
    for gene in dict_eQTL.keys():
        df_gene = pd.DataFrame(dict_eQTL[gene].items(), columns=['Locus', 'P_value'])
        df_filtered = df_gene[df_gene['P_value'] > p_value_threshold]
        if len(df_filtered) > 0:
            res[gene] = df_filtered
    return res


def add_to_df_cis_or_trans(df_gene: pd.DataFrame , gene_loc: pd.DataFrame ,genotypes_df: pd.DataFrame) -> pd.DataFrame:
    range_size = 2*10**6    # 2Mbp

    # add gene location to snp df
    merged_snp_gene_loc = df_gene.assign(**gene_loc.iloc[0])

    # merge df1 with df2 to get SNP location
    merged = pd.merge(merged_snp_gene_loc, genotypes_df, on='Locus',how="inner")

    merged['Cis\Trans'] = 'Trans'
    same_chromosome_mask = merged['representative genome chromosome'] == merged['Chr_Build37']
    merged.loc[same_chromosome_mask, 'Cis\Trans'] = ''
    cis_mask = same_chromosome_mask & \
               ((merged['representative genome start'] - range_size) <= merged['Build37_position']) & \
               (merged['Build37_position'] <= (merged['representative genome end'] + range_size))
    merged.loc[cis_mask, 'Cis\Trans'] = 'Cis'
    merged.loc[~cis_mask, 'Cis\Trans'] = 'Trans'

    return merged[["Locus", "P_value", 'Cis\Trans']]


def get_gene_location(df: pd.DataFrame, gene: str) -> pd.DataFrame:
    selected_gene = df[df['marker symbol'] == gene][['representative genome chromosome', 'representative genome start','representative genome end']]
    selected_gene = selected_gene.astype(int)
    # merged['representative genome chromosome'] = merged['representative genome chromosome'].astype(int)
    return selected_gene
    # return selected_gene.iloc[0]['representative genome chromosome'], selected_gene.iloc[0]['representative genome start'], selected_gene.iloc[0]['representative genome end']


def plot_num_genes_per_eQTL(genes_to_snps:dict):
    snp_dict = {}
    for gene, df in genes_to_snps.items():
        snps = df['Locus']  # Replace with the actual column name containing SNPs
        for snp in snps:
            if snp not in snp_dict:
                snp_dict[snp] = set()
            snp_dict[snp].add(gene)

    for snp, genes in snp_dict.items():
        snp_dict[snp]=len(genes)

    genotype_df = pd.read_excel("genotypes.xls", header=1)[["Locus", "Chr_Build37",	"Build37_position"]]
    # Add a new column for SNP counts
    genotype_df['SNP_Count'] = genotype_df["Locus"].map(snp_dict).fillna(0)
    genotype_df['Locus'] = genotype_df['Locus'].astype(str)

    genotype_df_filtered = genotype_df[genotype_df["Locus"].isin(snp_dict.keys())]
    grouped = genotype_df.groupby('Chr_Build37')


    # Plotting
    fig, ax = plt.subplots(figsize=(100, 6))

    for group, data_group in grouped:
        x = data_group['Locus']
        y = data_group['SNP_Count']
        ax.bar(x, y, label=group)

    ax.set_xlabel('Locus')
    ax.set_ylabel('Gene Count')
    ax.set_title('Gene Count Per eQTL genome Wide (Chromose and Lucos)')
    ax.legend()

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def analyse_significant_eQTLs(genes_to_snps: dict):
    snp_dict = {}

    # Loop through each gene and extract SNP information
    for gene, df in genes_to_snps.items():
        snps = df['Locus']  # Replace with the actual column name containing SNPs
        cis_or_trans = df['Cis\Trans']  # Replace with the actual column name containing significance
        for snp, act in zip(snps, cis_or_trans):
            if snp not in snp_dict:
                snp_dict[snp] = set()
            snp_dict[snp].add(act)

    cis_count = 0
    trans_count = 0
    both_count = 0

    # Loop through the dictionary and count SNPs based on significance
    for act in snp_dict.values():
        if 'Cis' in act and 'Trans' in act:
            both_count += 1
        elif 'Cis' in act:
            cis_count += 1
        elif 'Trans' in act:
            trans_count += 1

    print(f"Different sifnificant eQTLS count: {len(snp_dict)}")
    print(f"Only 'Cis' SNPs: {cis_count}")
    print(f"Only 'Trans' SNPs: {trans_count}")
    print(f"Both 'Cis' and 'Trans' SNPs: {both_count}")


def get_gene_boundries(df:pd.DataFrame,gene:str)->(int,int):
    return df[df['marker symbol']==gene][['representative genome start','representative genome end']]

P_VALUE_THREASHOLD = 4.667  # TODO: show how we calculated
def analyse_eQTL_dict():
    # open csv for gene boundries
    MGI_Coordinates_df = pd.read_csv("MGI_Coordinates.Build37.rpt.txt", sep="\t")
    genotype_df = pd.read_excel("genotypes.xls", header=1)[["Locus", "Chr_Build37",	"Build37_position"]]

    # open pickel file (with genes to snps and pvalues)
    with open('eqtl_res_dict.pickle', 'rb') as f:
        eqtl_dict = pickle.load(f)

    # filter SNPs from each gene
    genes_relevant_snps = filter_weak_associated_genes(eqtl_dict, P_VALUE_THREASHOLD)

    for gene in genes_relevant_snps.keys():
        gene_loc = get_gene_location(MGI_Coordinates_df, gene)
        genes_relevant_snps[gene] = add_to_df_cis_or_trans(genes_relevant_snps[gene], gene_loc, genotype_df)

    analyse_significant_eQTLs(genes_relevant_snps)


def marker_to_gene_vissualization():
    # open csv for gene boundries
    lps_sim = pd.read_csv("dendritic_LPS_stimulation.txt", sep="\t")
    genotype_df = pd.read_excel("genotypes.xls", header=1)[["Locus", "Chr_Build37",	"Build37_position"]]

    # open pickel file (with genes to snps and pvalues)
    with open('eqtl_res_dict.pickle', 'rb') as f:
        eqtl_dict = pickle.load(f)

    # filter SNPs from each gene
    genes_relevant_snps = filter_weak_associated_genes(eqtl_dict, P_VALUE_THREASHOLD)

    genotype_df.sort_values(['Chr_Build37','Build37_position'],inplace = True)
    genes = lps_sim['data']
    snps = genotype_df['Locus']

    df = pd.DataFrame(0, index=snps, columns=genes)

    for gene , snp_df in tqdm(genes_relevant_snps.items()):
        for snp in snp_df['Locus']:
            df.loc[snp,gene]=1

    # Create a scatter plot
    fig, ax = plt.subplots()

    for col in df.columns:
        for idx, value in enumerate(df[col]):
            if value == 1:
                ax.scatter(col, df.index[idx], color='red', marker='o')

    # Set labels for axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    pass