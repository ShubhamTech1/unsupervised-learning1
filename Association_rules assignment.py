'''
# Data Mining Unsupervised Learning / Descriptive Modeling - Association Rule Mining

# Problem Statement:
    Q1. Kitabi Duniya, a famous bookstore in India, was established before Independence, the growth of the company was incremental
    year by year, but due to the online selling of books and widespread Internet access, its annual growth started to
    collapse. As a Data Scientist, you must help this heritage bookstore gain its popularity back    and increase the 
    footfall of customers and provide ways to improve the business exponentially to an expected value at a 25% improvement 
    of the current rate. Apply the pattern mining techniques (Association Rules Algorithm) to identify ways to improve sales. 
    Explain the rules (patterns) identified, and visually represent the rules in graphs for a clear understanding of the solution.
    


# CRISP-ML(Q) process model describes six phases:
# 
# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Model Evaluation
# 5. Deployment
# 6. Monitoring and Maintenance
'''

'''
1st STEP:
1. Business and Data Understanding :
'''
# Objective(s): Increase the Revenue, increase sales
# Constraints: minimizing the risk of customer alienation.


'''Success Criteria'''

# Business Success Criteria:  Achieve a 25% improvement in the current annual revenue.

# ML Success Criteria: Achieve a high level of accuracy in associations in the transaction data using the Association Rules Algorithm. 

# Economic Success Criteria: Reduce operational costs


'''
data collection
'''
# dataset of books is availbale in our lMS website.
#  - dataset contain 2000 customer who buying some books
#  - and 11 books name is recorded for each customer
                   
# data description : 
#    - we have 11 different different books in store.
#   - books name (Features) like childBks, YouthBks,CookBks,DoItYBks,RefBks,ArtBks,GeogBks,ItalCook,ItalAtlas,ItalArt,Florence 
#     this is all different different books are present in our stored.  

        
  
'''
2nd STEP: 
Data preparation (data cleaning)    
'''
import pandas as pd
book_df = pd.read_csv(r"D:\DATA SCIENTIST\DATA SCIENCE\DATASETS\book.csv")

# Credentials to connect to sql Database
from sqlalchemy import create_engine
user = 'root'  # user name
pw = 'root'  # password
db = 'book_db'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# to_sql() - function to push the dataframe onto a SQL table.
book_df.to_sql('book_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from book_tbl;'
org_df = pd.read_sql_query(sql, engine)


org_df.columns
org_df.dtypes
org_df.shape
org_df.info()
org_df.duplicated().sum()
df = org_df.drop_duplicates()
df.duplicated().sum()
df.info()



# outlier treatment:
# now we want to see if any outliers are present or not.
df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
# not any outliers are present in this dataframe 


# our data is already in standard format (numerical format)so, we dont need to apply normalization and standardization technique here



### Elementary Analysis ###
# Most popular items:
count = df.loc[:, :].sum() 

# Generates a series
pop_item = count.sort_values( ascending = False).head(11)

# Convert the series into a dataframe 
pop_item = pop_item.to_frame() # type casting

# Reset Index
pop_item = pop_item.reset_index()
pop_item

pop_item = pop_item.rename(columns = {"index": "items", 0: "count"})
pop_item

# here we can say 218 customers buying CookBks book, this book is most popular book in store


# Data Visualization
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 6) # rc stands for runtime configuration 
plt.style.use('dark_background')
pop_item.plot.barh()  # horizontal bar plot
plt.title('Most popular items')
plt.gca().invert_yaxis() # gca means "get current axes"




'''
3rd step:
Model Building (data mining)    
'''
# support :
from mlxtend.frequent_patterns import apriori, association_rules

# Itemsets
frequent_itemsets = apriori(df, min_support = 0.0075, max_len = 4, use_colnames = True)
frequent_itemsets

# Most frequent itemsets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)
frequent_itemsets



# Association Rules:
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(10)

rules.sort_values('lift', ascending = False).head(10)

# save this model
import pickle 

with open('association_rule_model.pkl', 'wb') as file:
    pickle.dump(rules, file)
    
import os
os.getcwd()








# remove duplicates:
# Handling Profusion of Rules (Duplication elimination)
def to_list(i):
    return (sorted(list(i)))

# Sort the items in Antecedents and Consequents based on Alphabetical order
ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

# Sort the merged list of items - transactions
ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

# No duplication of transactions
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]







# Capture the index of unique item sets
index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
    
index_rules

# Rules without any redudancy 
rules_no_redundancy = rules.iloc[index_rules, :]
rules_no_redundancy


# Sorted list and top 10 rules 
rules10 = rules_no_redundancy.sort_values('lift', ascending = False).head(10)
rules10 


'''
# here we can say, 
if we want to maximize our sales, then  we consider promoting items with high lift and confidence values together,
as they are more likely to be purchased as a bundle of 4 books.

we can see that certain combinations of books, like 'ItalAtlas' and 'ArtBks', 
have a high lift and confidence, indicating a strong association. we can use these insights to design marketing strategies,
recommendations or discounts to encourage customers to buy these book combinations, to increasing revenue.


'''



rules10.plot(x = "support", y = "confidence", c = rules10.lift, 
             kind = "scatter", s = 12, cmap = plt.cm.coolwarm)







#save file:
# Store the rules on to csv file
# csv do not accepting frozensets thats why we converted into string datatype. 

# Removing frozenset from dataframe
rules10['antecedents'] = rules10['antecedents'].astype('string')
rules10['consequents'] = rules10['consequents'].astype('string')

rules10['antecedents'] = rules10['antecedents'].str.removeprefix("frozenset({")
rules10['antecedents'] = rules10['antecedents'].str.removesuffix("})")

rules10['consequents'] = rules10['consequents'].str.removeprefix("frozenset({")
rules10['consequents'] = rules10['consequents'].str.removesuffix("})")


rules10.to_csv("kitabiduniya.csv",encoding = 'utf-8')















