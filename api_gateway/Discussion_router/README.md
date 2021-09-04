# FILE TREE

\
|--- portal/
        |---
        |---
        |---
|--- api_gateway/
        |--- main.py    ---> API server
        |--- Discussion_router/
                    |--- Agreement3_full_df.pkl           
                        --> pickle file consisting of Agreement model(sentence level)                     
                    |--- Disagreement1_full_df.pkl       
                        --> pickle file consisting of Disagreement model(sentence level)                     
                    |--- Partial2_full_df.pkl             
                        --> pickle file consisting of Partial model(sentence level)                      
                    |--- branchlevel_model.pkl            
                        --> pickle file consisting of branch level model                     
                    |--- vectorizer_tfidf_fulldata.pkl    
                        --> pickle file consisting of tfidf vectorizer for sentence level                            
                    |--- vectorizer.pk                    
                        --> pickle file consisting of dense tfidf vectorizer for branch level            
                    |--- branch.py                        
                        --> Used to get branch level result, used in userdata.py          
                    |--- discussion.py                    
                        --> API router              
                    |--- fun.py                           
                        --> Used to get the intervention result and branch level result, used in discussion.py     
                    |--- train.py                         
                        --> Training script that uses old data and new data from db to retrain model         
                    |--- userdata.py                      
                        --> Script to get suitable interveners in sorted order of their score             
                    |--- util.py                          
                        --> Useful dictionaries used for preprocessing. 
                    |--- segregated_formdata_comment.xls  
                        --> old data used for training the model in train.py
                    |--- Retrained/
                            |--- ... --> the pickle files created by train.py is stored here
        |--- ...  --> other existing files
