import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log10
import statsmodels.api as sm
from scipy.stats import t, f, f_oneway
import pickle
from tqdm import tqdm

NON_DATA_COLS = ["ID_FOR_CHECK", "Phenotype", "Authors", "Year", "Pubmed Id", "empty_cnt", "std"]


# The following funcs are general and used for Q1 and Q2
def calc_beta1(x_data, y_data):
    x_mean = np.mean(x_data).item()
    y_mean = np.mean(y_data).item()
    numerator = 0
    denominator = 0
    for i in range(len(x_data)):
        numerator += (x_data.iloc[i].item() - x_mean) * (y_data.iloc[i].item() - y_mean)
        denominator += ((x_data.iloc[i].item() - x_mean) ** 2)
    return numerator / denominator


def calc_beta0(x_data, y_data, beta1):
    return np.mean(y_data).item() - beta1*(np.mean(x_data).item())


def calc_sse(y_data, y_pred):
    sse = 0
    for i in range(len(y_data)):
        sse += ((y_data.iloc[i].item() - y_pred.iloc[i].item()) ** 2)
    return sse


def calc_ssr(y_data, y_pred):
    y_mean = np.mean(y_data).item()
    ssr = 0
    for i in range(len(y_pred)):
        ssr += ((y_pred.iloc[i].item() - y_mean) ** 2)
    return ssr


def calc_t(beta, sse, x_data):
    p1 = sse / (len(x_data) - 2)
    p2 = 0
    x_mean = np.mean(x_data).item()
    for i in range(len(x_data)):
        p2 += ((x_data.iloc[i].item() - x_mean) ** 2)

    return beta / ((p1 / p2) ** 0.5)


def prep_genotype_data(df: pd.DataFrame, genotypes_to_consider: list):
    gn_map = {v: i for i, v in enumerate(genotypes_to_consider)}
    df["genotype"] = df["genotype"].map(gn_map)


# related to Q1
def find_relevant_phenotype():
    ph_df = pd.read_excel("phenotypes.xls")
    gn_df = pd.read_excel("genotypes.xls", header=1)
    gn_df = gn_df.rename(columns={0: "ID_FOR_CHECK"})

    for index, row in ph_df.iterrows():
        empty_cnt = 0
        data_vals = []
        for c in ph_df.columns:
            if c not in NON_DATA_COLS:
                if pd.isna(row[c]):
                    empty_cnt += 1
                else:
                    data_vals.append(row[c])

        ph_df.at[index, 'empty_cnt'] = empty_cnt
        ph_df.at[index, 'std'] = np.std(data_vals)

    # print(ph_df[ph_df["Pubmed Id"].notna()].sort_values(by=["empty_cnt", "std"], ascending=[True, False])[["ID_FOR_CHECK", "empty_cnt", "std", "Pubmed Id"]].head(50))
    ids = ph_df[ph_df["Pubmed Id"].notna()].sort_values(by=["empty_cnt", "std"], ascending=[True, False])[["ID_FOR_CHECK"]]
    rids = [i[0] for i in ids.values]
    fgn = gn_df[gn_df["ID_FOR_CHECK"].isin(rids)]
    rel_rows = {}
    for i, r in fgn.iterrows():
        for c in fgn.columns:
            if r[c] == "H":
                if not np.isnan(ph_df[ph_df["ID_FOR_CHECK"] == r["ID_FOR_CHECK"]][c].item()):
                    # print(r)
                    ec = ph_df[ph_df["ID_FOR_CHECK"] == r["ID_FOR_CHECK"]]["empty_cnt"].item()
                    if ec in rel_rows:
                        rel_rows[ec].append(r)
                    else:
                        rel_rows[ec] = [r]

    sum_df = pd.DataFrame({"empty_cnt": rel_rows.keys(), "ids": [[r["ID_FOR_CHECK"] for r in r_ls] for r_ls in rel_rows.values()]})
    print(sum_df.sort_values(by="empty_cnt", ascending=True).head())
    # print(f"lowest empty cnt: {min(rel_rows.keys())}")
    # print(f"phenotype ids with this cnt: {rel_rows[min(rel_rows.keys())]}")


WANTED_ID = 1195


# related to Q1
def regression_my_imp(x_data, y_data):
    beta1 = calc_beta1(x_data, y_data)
    beta0 = calc_beta0(x_data, y_data, beta1)
    y_pred = beta0 + beta1 * x_data
    sse = calc_sse(y_data, y_pred)
    ssr = calc_ssr(y_data, y_pred)
    sst = sse + ssr
    r_squared = ssr / sst
    deg_freedom = 2
    f_val = r_squared / ((1 - r_squared) / (len(x_data) - deg_freedom))
    p_val_f = 1 - f.cdf(f_val, 1, len(x_data) - deg_freedom)
    t_val = calc_t(beta1, sse, x_data)
    p_val_t = 2 * (1 - t.cdf(t_val, len(x_data) - deg_freedom))
    print(f"beta1 = {beta1}")
    print(f"beta0 = {beta0}")
    print(f"y = {beta0} + {beta1} * x")
    print(f"r^2 = {r_squared}")
    print(f"f-value = {f_val}")
    print(f"f-test p-value = {p_val_f}")
    print(f"t-value = {t_val}")
    print(f"t-test p-value regarding beta1 = {p_val_t}")


# related to Q1 (to verify my implementation)
def run_regression(x_data, y_data):
    X = sm.add_constant(x_data)
    model = sm.OLS(y_data, X).fit()

    # Plotting the data points
    plt.scatter(x_data, y_data, label='Data')

    # Generating x values for the regression line
    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    # x_line_encoded = pd.get_dummies(x_line, drop_first=True)
    X_line = sm.add_constant(x_line)

    # Predicting y values for the regression line
    y_line = model.predict(X_line)

    # Plotting the regression line
    plt.plot(x_line, y_line, color='red', label='Regression Line')

    # Adding labels and legend to the plot
    plt.xlabel('Genotype')
    plt.ylabel('Phenotype')
    plt.legend()

    print(model.summary())
    print("Coefficients:")
    print(model.params)
    print("\nP-values:")
    print(model.pvalues)

    # Display the plot
    plt.show()


# related to Q1
def plot_data(df, genotypes_to_consider: list):
    df['genotype'] = pd.Categorical(df['genotype'], categories=genotypes_to_consider, ordered=True)

    # Sort the DataFrame based on the order of the 'Genotype' column
    df = df.sort_values('genotype')

    # Set up the plot
    fig, ax = plt.subplots()

    # Plot the data
    ax.scatter(df['genotype'], df['phenotype'])

    # Set the X-axis labels and order
    ax.set_xticks(range(len(genotypes_to_consider)))
    ax.set_xticklabels(genotypes_to_consider)

    # Set the Y-axis label
    ax.set_ylabel('Phenotype')
    plt.xlabel('Genotype')
    plt.title("Phenotype vs Genotype")
    plt.show()


# related to Q1
def prepare_data(genotypes_to_consider: list):
    gn_df = pd.read_excel("genotypes.xls", header=1)
    gn_df = gn_df.rename(columns={0: "ID_FOR_CHECK"})
    ph_df = pd.read_excel("phenotypes.xls")

    gn_row = gn_df[gn_df["ID_FOR_CHECK"] == WANTED_ID]
    ph_row = ph_df[ph_df["ID_FOR_CHECK"] == WANTED_ID]

    ph_non_empty_breeds = {}
    for c in ph_df.columns:
        if c not in NON_DATA_COLS:
            if not ph_row[c].isna().item():
                ph_non_empty_breeds[c] = ph_row[c].item()

    # ignore H
    gn_filtered_breeds = {}
    for br in ph_non_empty_breeds:
        if gn_row[br].item() in genotypes_to_consider:
            gn_filtered_breeds[br] = gn_row[br].item()

    df_dict = {"breed": [], "genotype": [], "phenotype": []}
    for br in gn_filtered_breeds:
        df_dict["breed"].append(br)
        df_dict["genotype"].append(gn_filtered_breeds[br])
        df_dict["phenotype"].append(ph_non_empty_breeds[br])

    df = pd.DataFrame(df_dict)
    return df


# related to Q1
def regression_tests(genotypes_to_consider: list):
    df = prepare_data(genotypes_to_consider)

    # plot_data(df, genotypes_to_consider)

    # run_regression(pd.get_dummies(df["genotype"], drop_first=True), df["phenotype"])
    prep_genotype_data(df, genotypes_to_consider)
    regression_my_imp(df["genotype"], df["phenotype"])


# related to Q1
def run_anova(df, genotypes_to_consider: list):
    # Group the phenotypes by genotype
    genotype_groups = [df['phenotype'][df['genotype'] == g] for g in genotypes_to_consider]

    # Perform one-way ANOVA
    f_value, p_value = f_oneway(*genotype_groups)

    print(f"ANOVA:")
    print("F-value:", f_value)
    print("p-value:", p_value)


# related to Q1
def anova_my_imp(df, genotypes_to_consider: list):
    phen_mean = df['phenotype'].mean()
    genotype_groups = [df['phenotype'][df['genotype'] == g] for g in genotypes_to_consider]
    ss_among = sum([len(g)*((g.mean() - phen_mean)**2) for g in genotype_groups])
    ss_within = sum([sum([(i - g.mean())**2 for i in g]) for g in genotype_groups])
    deg_freedom_among = len(genotype_groups) - 1
    deg_freedom_within = sum([len(g) for g in genotype_groups]) - len(genotype_groups)
    ms_among = ss_among / deg_freedom_among
    ms_within = ss_within / deg_freedom_within
    f_val = ms_among / ms_within
    p_val = 1 - f.cdf(f_val, deg_freedom_among, deg_freedom_within)
    print(f"F-value: {f_val}")
    print(f"p-value: {p_val}")


# related to Q1
def anova_test(genotypes_to_consider: list):
    df = prepare_data(genotypes_to_consider)

    # run_anova(df, genotypes_to_consider)
    anova_my_imp(df, genotypes_to_consider)


# related to Q2
def regression_model(x_data, y_data):
    """
    Gets x_data and y_data already cleaned and ordered
    performs linear regression model
    calcs p-value based on F-test
    returns -log(p-value)
    :param x_data:
    :param y_data:
    :return:
    """
    beta1 = calc_beta1(x_data, y_data)
    beta0 = calc_beta0(x_data, y_data, beta1)
    y_pred = beta0 + beta1 * x_data
    sse = calc_sse(y_data, y_pred)
    ssr = calc_ssr(y_data, y_pred)
    sst = sse + ssr
    r_squared = ssr / sst
    deg_freedom = 2
    f_val = r_squared / ((1 - r_squared) / (len(x_data) - deg_freedom))
    p_val_f = 1 - f.cdf(f_val, 1, len(x_data) - deg_freedom)
    return -log10(p_val_f)


# related to Q2
def prep_data(breed_gn: dict, breed_ph: dict):
    """
    Gets dict with breed to genotype and dict with breed to ph
    walks through the breeds with genotypes and creates a df with breed, genotype and phenotype
    returns the df
    :param breed_gn:
    :param breed_ph:
    :return:
    """
    df_dict = {"breed": [], "genotype": [], "phenotype": []}
    # breed gn may contain less breeds than breed_ph by earlier construction
    # so it will be used to prep the data
    for br in breed_gn:
        df_dict["breed"].append(br)
        df_dict["genotype"].append(breed_gn[br])
        df_dict["phenotype"].append(breed_ph[br])

    df = pd.DataFrame(df_dict)
    return df


def generate_qtl_dict(phenotype_ids: list, genotypes_file_path, phenotypes_file_path):
    qtl_dict = {}
    for phenotype_id in tqdm(phenotype_ids):
        snps_to_p_val = q_2_analysis(genotypes_file_path, phenotypes_file_path, phenotype_id, f"{phenotype_id}.csv")
        qtl_dict[phenotype_id] = snps_to_p_val


    with open("phenotypes_qtl_dict.pickle", 'wb') as f:
        pickle.dump(qtl_dict, f)


# related to Q2
def q_2_analysis(genotypes_file_path, phenotypes_file_path, wanted_phenotype_id, res_path):
    """
    Gets genotypes and phenotypes file paths
    Runs linear regression model on each SNP regarding phenotype with ID: 1195
    Outputs for each SNP its -log(p-value)
    :param genotypes_file_path:
    :param phenotypes_file_path:
    :return:
    """
    # read the data
    gn_df = pd.read_excel(genotypes_file_path, header=1)
    gn_df = gn_df.rename(columns={0: "ID_FOR_CHECK"})
    ph_df = pd.read_excel(phenotypes_file_path)

    # get chosen phenotype
    ph_row = ph_df[ph_df["ID_FOR_CHECK"] == wanted_phenotype_id]

    # filter for breeds that contain phenotype value (not Nan)
    ph_non_empty_breeds = {}
    for c in ph_df.columns:
        if c not in NON_DATA_COLS:
            if not ph_row[c].isna().item():
                ph_non_empty_breeds[c] = ph_row[c].item()

    # ignore H
    # and build a map of snp to its genotype (with breeds that have phenotype value)
    genotypes_to_consider = ['B', 'D']
    snp_to_gn = {}
    for i, r in gn_df.iterrows():
        snp = r["Locus"]
        gn_filtered_breeds = {}
        for br in ph_non_empty_breeds:
            if br in r and r[br] in genotypes_to_consider:
                gn_filtered_breeds[br] = r[br]
        snp_to_gn[snp] = gn_filtered_breeds

    # run regression on each SNP and save the result which is -log(p-value)
    snp_to_res = {}     # res is -log(p-value)
    for snp in snp_to_gn:
        df = prep_data(snp_to_gn[snp], ph_non_empty_breeds)
        prep_genotype_data(df, genotypes_to_consider)
        res = regression_model(df["genotype"], df["phenotype"])
        snp_to_res[snp] = res

    # saves results to file
    res_df = pd.DataFrame({"snp": snp_to_res.keys(), "-log(p-value)": snp_to_res.values()})
    res_df.to_csv(res_path)
    return res_df


# related to Q2
def plot_q2_results(data_path, phenotype_name: str = None):
    """
    Gets path to the output file from q_2_analysis func
    Plots manhattan plot and prints best SNP
    :param data_path:
    :return:
    """
    df = pd.read_csv(data_path)
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
    plt.title(f'Manhattan Plot for: {phenotype_name}')
    plt.xticks(range(len(df)), df[snp_c_name], rotation=90)  # Show SNP names on x-axis
    # plt.tight_layout()
    plt.show()

    # Get the best-scored SNP
    sorted_df = df.sort_values(by=p_val_c_name, ascending=False)
    best_snp = sorted_df.iloc[0][snp_c_name]
    best_pvalue = sorted_df.iloc[0][p_val_c_name]
    print(f"The best-scored SNP is {best_snp} with a -log(p-value) of {best_pvalue}.")


if __name__ == '__main__':
    # regression_tests(genotypes_to_consider=['B', 'D'])
    # regression_tests(genotypes_to_consider=['B', 'H', 'D'])
    # anova_test(genotypes_to_consider=['B', 'D'])
    # find_relevant_phenotype()
    # q_2_analysis("genotypes.xls", "phenotypes.xls")
    # plot_q2_results("snps_p_val.csv")
    pass