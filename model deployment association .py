from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from mlxtend.frequent_patterns import apriori, association_rules
from sqlalchemy import create_engine

loaded_rules = pickle.load(open('association_rule_model.pkl','rb'))

# Connecting to sql by creating sqlachemy engine
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root", # user
                               pw = "root", # password
                               db = "book_db")) # database
# Define flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        
        data = pd.read_csv(f, sep=';', header=None)
        
        data = data.iloc[:, 0].to_list()
        
       # drop duplicated records from dataframe
                
        df = data.drop_duplicates()
        
                
        # Itemsets
              
        frequent_itemsets = apriori(df, min_support = 0.0075, max_len = 4, use_colnames = True)
        frequent_itemsets
        
                
        # Most frequent itemsets based on support 
        frequent_itemsets.sort_values('support', ascending = False, inplace = True)
        frequent_itemsets
        
        rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
        rules.head()
        
       
        def to_list(i):
            return (sorted(list(i)))

        ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

        # Sort the merged list of items - transactions
        ma_X = ma_X.apply(sorted)

        rules_sets = list(ma_X)

        unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

        
        index_rules = []
        for i in unique_rules_sets:
            index_rules.append(rules_sets.index(i))
            

        # Rules without any redudancy 
        rules_no_redundancy = rules.iloc[index_rules, :]
        rules_no_redundancy


        # Sorted list and top 10 rules 
        rules10 = rules_no_redundancy.sort_values('lift', ascending = False).head(15)
        rules10 = rules10.replace([np.inf, -np.inf], np.nan)
         
        rules10['antecedents'] = rules10['antecedents'].astype('string')
        rules10['consequents'] = rules10['consequents'].astype('string')
 
        rules10['antecedents'] = rules10['antecedents'].str.removeprefix("frozenset({")
        rules10['antecedents'] = rules10['antecedents'].str.removesuffix("})")
 
        rules10['consequents'] = rules10['consequents'].str.removeprefix("frozenset({")
        rules10['consequents'] = rules10['consequents'].str.removesuffix("})")
 
 
        rules10.to_csv("kitabiduniya.csv",encoding = 'utf-8')



        html_table = rules10.to_html(classes = 'table table-striped')
       
                
                
        return render_template("new.html", Y =   f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #39648f;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                        .table td {{\
                        background-color: #5e617d;\
                    }}\
                            .table tbody th {{\
                            background-color: #ab2c3f;\
                        }}\
                </style>\
                {html_table}") 
                
if __name__=='__main__':
    app.run(debug = False) 

