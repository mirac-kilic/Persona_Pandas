# Persona_Pandas
# Rule-Based Classification has been applied to the Persona.csv dataset to calculate the Potential Customer Revenue.

import pandas as pd

pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
pd.set_option('display.float_format',lambda x:"%.2f" %x)

# TASK 1
# 1. Read the persona.csv file and display general information about the dataset.
df=pd.read_csv("persona.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.info())
print("1-------------------------")


# 2. How many unique SOURCE values are there? What are their frequencies?
print(df["SOURCE"].nunique())
print(df["SOURCE"].value_counts())
print("2-------------------------")


# 3. How many unique PRICE values are there?
print(df["PRICE"].nunique())
print("3-------------------------")


# 4. How many sales have been made for each `PRICE`?
print(df["PRICE"].value_counts())
print("4-------------------------")


# 5. How many sales have been made from each country?
print(df["COUNTRY"].value_counts())
print(df.groupby("COUNTRY")["PRICE"].count())
print("5-------------------------")


# 6. How much total revenue has been generated from sales in each country?
print(df.groupby("COUNTRY").agg({"PRICE": "sum"}))
print(df["PRICE"].sum())
print("6-------------------------")


# 7. What are the sales numbers for each `SOURCE` type?
print(df.groupby("SOURCE").agg({"PRICE": "count"}))
print("7-------------------------")


# 8. What are the average `PRICE` values by country?
print(df.groupby("COUNTRY").agg({"PRICE": "mean"}))
print("8-------------------------")


# 9. What are the average `PRICE` values by `SOURCE`?
print(df.groupby("SOURCE").agg({"PRICE": "mean"}))
print("9-------------------------")


# 10. What are the average `PRICE` values broken down by `COUNTRY` and `SOURCE`?
print(df.groupby(["COUNTRY","SOURCE"]).agg({"PRICE": "mean"}))
print("10-------------------------")


# TASK 2: What are the average earnings broken down by `COUNTRY`, `SOURCE`, `SEX`, and `AGE`?
print(df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"}).head())
print("TASK 2-------------------------")


# TASK 3: Sort the output by `PRICE`.
agg_df=pd.DataFrame(df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"}).sort_values("PRICE",ascending=False))
print(agg_df.head())
print(type(agg_df))
print("TASK 3-------------------------")


# TASK 4: Convert the names in the index to variable names.
agg_df=agg_df.reset_index()
print(agg_df.columns)
print("TASK 4-------------------------")


# TASK 5: Convert the `Age` variable into a categorical variable and add it to `agg_df`.
print(agg_df["AGE"].describe())
agg_df["AGE_CAT"]=pd.cut(agg_df["AGE"],bins=[0,18,23,30,40,agg_df["AGE"].max()],
                         labels=["0_18","19_23","24_30","31_40","41_70"],
                         include_lowest=True)
print(agg_df.head(10))
print("TASK 5-------------------------")


# TASK 6: Define the new level-based customers (personas).
"""agg_df["customer_level_based"]=agg_df[["COUNTRY","SOURCE","SEX","AGE_CAT"]].agg(lambda x:"_".join(x).upper(),axis=1)
print(agg_df.head(10))"""
for row in agg_df.values:
    print(row)

agg_df["customer_level_based"]=[row[0].upper()+"_"+ row[1].upper() +"_"+ row[2].upper() +"_"+ row[5].upper() for row in agg_df.values]
agg_df=agg_df[["customer_level_based","PRICE"]]
print(agg_df.head())
for i in agg_df["customer_level_based"].values:
    print(i.split("_"))

print(agg_df["customer_level_based"].value_counts())
agg_df=agg_df.groupby("customer_level_based").agg({"PRICE": "mean"})
agg_df=agg_df.reset_index()
print(agg_df.head())
print(agg_df["customer_level_based"].value_counts())
print("TASK 6-------------------------")


# TASK 7: Segment the new customers (personas).
agg_df["SEGMENT"]=pd.qcut(agg_df["PRICE"],4,labels=["D","C","B","A"])
print(agg_df.tail(20))
print(agg_df.groupby("SEGMENT",observed=True).agg({"PRICE":["mean","max","sum"]}))
print(agg_df.head(20))
print("TASK 7-------------------------")


# TASK 8: Classify the new customers and predict how much revenue they could generate.
new="TUR_ANDROID_FEMALE_31_40"
print(agg_df[agg_df["customer_level_based"]==new])
new_2="FRA_IOS_FEMALE_31_40"
print(agg_df[agg_df["customer_level_based"]==new_2])
print(agg_df.loc[agg_df["customer_level_based"]==new,"PRICE"].values[0])
print("TASK 8-------------------------")





