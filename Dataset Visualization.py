import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

folder_path = 'Dataset Visualization'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Load the dataset
df = pd.read_csv('ADNI1_Complete_1Yr_1.5T.csv')

# Replace 'M' with 'Male' and 'F' with 'Female'
df['Sex'] = df['Sex'].replace({'M': 'Male', 'F': 'Female'})

# Rename the groups for better readability
group_names = {
    'AD': "Alzheimer's Disease (AD)",
    'MCI': "Mild Cognitive Impairment (MCI)",
    'CN': "Cognitively Normal (CN)"
}
df['Group'] = df['Group'].map(group_names)

# Define the order of the groups
group_order = ["Alzheimer's Disease (AD)", "Mild Cognitive Impairment (MCI)", "Cognitively Normal (CN)"]

# Set the style for a professional look
sns.set_style("whitegrid")

# ----------------------------
# 1. Sex Distribution
# ----------------------------
plt.figure(figsize=(8, 6))  # Set the figure size
sns.countplot(x='Sex', data=df, palette={'Male': 'blue', 'Female': 'pink'}, edgecolor='black')
plt.title('Distribution of Subjects by Sex', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Sex', fontsize=14, fontweight='bold')
plt.ylabel('Number of Subjects', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine()
plt.tight_layout()
plt.savefig('Dataset Visualization/Sex Distribution.png')
plt.show()

# ----------------------------
# 2. Age Distribution
# ----------------------------
plt.figure(figsize=(8, 6))  # Set the figure size
sns.histplot(df['Age'], bins=30, kde=True, color='green', edgecolor='black')
plt.title('Distribution of Age Among Subjects', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Age (Years)', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine()
plt.tight_layout()
plt.savefig('Dataset Visualization/Age Distribution.png')
plt.show()

# ----------------------------
# 3. Group Distribution
# ----------------------------
plt.figure(figsize=(8, 6))  # Set the figure size
sns.countplot(
    x='Group',
    data=df,
    order=group_order,  # Specify the order of the groups
    palette={"Alzheimer's Disease (AD)": 'red', "Mild Cognitive Impairment (MCI)": 'orange', "Cognitively Normal (CN)": 'green'}
)
plt.title('Distribution of Subjects by Diagnostic Group', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Diagnostic Group', fontsize=14, fontweight='bold')
plt.ylabel('Number of Subjects', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine()
plt.tight_layout()
plt.savefig('Dataset Visualization/Group Distribution.png')
plt.show()