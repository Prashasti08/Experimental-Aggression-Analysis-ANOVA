import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Data Loading
df = pd.read_csv(("/Users/PT/2025 fall/programming for data science/final project/plosone.csv")) 

# Checking Number of Rows in Dataset
num_rows = df.shape[0]
print(num_rows)

# Defining Relevant Columns 
key_cols = ['gender', 'Buss-Perry Aggression Questionnaire total score', 'Block 1 mean', 'Block 2 mean', 'Block 3 mean']
df_relevant = df[key_cols].copy()

# Checking Missing Values
print("Missing Values:\n", df_relevant.isnull().sum())


# Data Type Identification and Adjustment
df_relevant['gender'] = df_relevant['gender'].astype('category')
print(df_relevant['gender'].dtype)


# Outlier Detection and Removal
numerical_cols = ['Buss-Perry Aggression Questionnaire total score', 'Block 1 mean', 'Block 2 mean', 'Block 3 mean']
df_clean = df_relevant.copy() 

initial_rows = df_clean.shape[0]

for col in numerical_cols:
    # Calculate IQR Bounds
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    # Filter: Keep only the rows WITHIN the bounds
    filter_keep = (df_clean[col] >= lower) & (df_clean[col] <= upper)
    
    # Update df_clean by removing outliers
    df_clean = df_clean[filter_keep].copy()

#  Final Check
print("\nRows Removed:", initial_rows - df_clean.shape[0])
print("Final Row Count:", df_clean.shape[0])
print("Cleaning complete. Data is in 'df_clean'.")


# Defining the columns containing the block scores
block_cols = ['Block 1 mean', 'Block 2 mean', 'Block 3 mean']

# Sample Size by Gender
print("--- 1. Sample Size by Gender ---")

# Counting the number of rows for each unique value in the 'gender' column
print(df_clean['gender'].value_counts())
print("\n")

# Calculating Mean and SD of BPAQ scores for each gender
print("--- 2. BPAQ Mean and Standard Deviation by Gender ---")

# Grouping data by gender, then calculating mean and SD for the BPAQ score
bpaq_stats = df_clean.groupby('gender')['Buss-Perry Aggression Questionnaire total score'].agg(['mean', 'std'])
print(bpaq_stats)
print("\n")

# Overall Mean Aggression Scores for Blocks 1, 2, and 3
print("--- 3. Overall Mean Aggression Scores (Across All Participants) ---")
overall_means = df_clean[block_cols].mean()
print(overall_means)
print("\n")

#####################################################
# After we done the data processing part here we start Statistical Analysis
class RQ1Stat:
    """
    This class contains statistical methods that we need to finish rq1.

    """
    def __init__(self,data:pd.DataFrame):
        self.data = data
    
    def split_by_gender(self):
        male = self.data.loc[self.data["gender"] == "male", "Buss-Perry Aggression Questionnaire total score"].dropna()
        female = self.data.loc[self.data["gender"] == "female", "Buss-Perry Aggression Questionnaire total score"].dropna()
        return {
       "male_mean": male.mean(),
       "female_mean": female.mean(),
       "male_sd": male.std(),
       "female_sd": female.std()
       }
    
    def gender_bpaq_ols(self):
        """
        After we split the gender group we should compute the t test using ols
        """
        model = smf.ols(
            'Q("Buss-Perry Aggression Questionnaire total score") ~ C(gender)',
            data=self.data
        ).fit()
        return model
    
print("--- RQ1 Descriptive Statistics: BPAQ by Gender ---")
rq1_desc = df_clean.groupby("gender", observed=False)[
    "Buss-Perry Aggression Questionnaire total score"
].agg(["mean", "std", "count"])
print(rq1_desc)
print("\n")
stats = RQ1Stat(df_clean)
rq1_model = stats.gender_bpaq_ols()
print(rq1_model.summary())

# Here We start RQ2
data_rq2 = pd.melt(
    df_clean,
    id_vars=["gender"],
    value_vars=["Block 1 mean", "Block 2 mean", "Block 3 mean"],
    var_name="block",
    value_name="aggression"
)

rq2_model = smf.ols(
    "aggression ~ C(gender) * C(block)",
    data=data_rq2
).fit()

anova_table = sm.stats.anova_lm(rq2_model, typ=2)
print("--- RQ2 Descriptive Statistics: Mean Aggression by Block & Gender---")
block_means = data_rq2.groupby("block")["aggression"].agg(["mean", "std", "count"])
print(block_means)
print("\n")
print(anova_table)

#####################################################
# ggplot-style
class GGPlotStyle:
    """
    Provides a reusable ggplot-like grayscale theme
    for all figures in the study.
    """

    def __init__(self):
        self.palette = {
            "male": "#7F7F7F",
            "female": "#BFBFBF"
        }

    def apply(self):
        sns.set_theme(
            style="whitegrid",
            font_scale=1.1,
            rc={
                "grid.color": "#E5E5E5",
                "grid.linewidth": 1,
                "axes.edgecolor": "black",
                "axes.spines.top": False,
                "axes.spines.right": False
            }
        )

    def colors(self):
        return self.palette

## apply ggplot-like style
style = GGPlotStyle()
style.apply()
palette = style.colors()

# Figure 1: BPAQ scores by gender
plt.figure(figsize=(6, 5))

sns.boxplot(
    data=df_clean,
    x="gender",
    y="Buss-Perry Aggression Questionnaire total score",
    palette=palette,
    width=0.5
)

sns.stripplot(
    data=df_clean,
    x="gender",
    y="Buss-Perry Aggression Questionnaire total score",
    color="black",
    alpha=0.35,
    jitter=0.15,
    size=3
)

plt.xlabel("Gender")
plt.ylabel("BPAQ Total Score")
plt.show()

# Figure 2: Aggression across provocation levels by gender 
summary = (
    data_rq2
    .groupby(["gender", "block"])
    .agg(
        mean_aggr=("aggression", "mean"),
        se_aggr=("aggression", lambda x: x.std() / (len(x) ** 0.5))
    )
    .reset_index()
)

block_map = {
    "Block 1 mean": "Low",
    "Block 2 mean": "Medium",
    "Block 3 mean": "High"
}

summary["Provocation"] = summary["block"].map(block_map)
summary["Provocation"] = pd.Categorical(
    summary["Provocation"],
    categories=["Low", "Medium", "High"],
    ordered=True
)

plt.figure(figsize=(7, 5))

sns.lineplot(
    data=summary,
    x="Provocation",
    y="mean_aggr",
    hue="gender",
    palette=palette,
    marker="o"
)

plt.xlabel("Provocation Level")
plt.ylabel("Mean Aggressive Responses")
plt.legend(title="Gender")
plt.show()

# Figure 3: Trait aggression × provocation × gender
fig3_data = pd.melt(
    df_clean,
    id_vars=[
        "gender",
        "Buss-Perry Aggression Questionnaire total score"
    ],
    value_vars=[
        "Block 1 mean",
        "Block 2 mean",
        "Block 3 mean"
    ],
    var_name="block",
    value_name="aggression"
)

block_map = {
    "Block 1 mean": "Low",
    "Block 2 mean": "Medium",
    "Block 3 mean": "High"
}

fig3_data["Provocation"] = fig3_data["block"].map(block_map)
fig3_data["Provocation"] = pd.Categorical(
    fig3_data["Provocation"],
    categories=["Low", "Medium", "High"],
    ordered=True
)

fig3_data["gender"] = pd.Categorical(
    fig3_data["gender"],
    categories=["male", "female"],
    ordered=True
)

sns.lmplot(
    data=fig3_data,
    x="Buss-Perry Aggression Questionnaire total score",
    y="aggression",
    row="gender",
    col="Provocation",
    scatter_kws={"color": "black", "alpha": 0.45},
    line_kws={"color": "black", "linewidth": 2}
)

plt.show()