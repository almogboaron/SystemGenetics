import pandas as pd
from ex2.hw2_208273672 import regression_model, prep_data, prep_genotype_data
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os


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


PLOT_FOLDER = r"C:\Users\User1\Desktop\stuff\TAU\TASHPC\sem_b\systems_genetics\ex3\plots"


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


def association_test():
    # load genotypes
    gen_df = pd.read_excel("genotypes.xls", header=1)

    # we want to filter the neighbour locis with the same genotype
    columns = gen_df.columns.delete([0, 1, 2, 3])
    gen_df = filter_neighboring_rows(gen_df, columns)

    # load expresion data
    lps_df = pd.read_csv("dendritic_LPS_stimulation.txt", sep="\t")
    exp_data = add_together_same_bxd(lps_df)
    exp_data.drop(columns=["B6", "D2"], inplace=True)
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

    with open("eqtl_res_dict.pickle", 'wb') as f:
        pickle.dump(eqtl_res, f)

    print("finish")


def understand_task():
    gen_df = pd.read_excel("genotypes.xls", header=1)
    lps_df = pd.read_csv("dendritic_LPS_stimulation.txt", sep="\t")

    gen_df.sort_values(by='Build37_position', inplace=True)

    # Find neighboring loci with consecutive "Build37_position" values
    neighboring_loci = []
    current_position = None

    for index, row in gen_df.iterrows():
        current_pos = row['Build37_position']
        if index < len(gen_df) - 1:
            if gen_df.loc[index + 1]["Build37_position"] == current_pos + 1:
                neighboring_loci.append(row)

    # Print the adjacent rows representing neighboring loci
    for loci_row in neighboring_loci:
        print(loci_row)


if __name__ == '__main__':
    # understand_task()
    association_test()