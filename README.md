# flick-pick

## Structure

.
├── data/                   
│   ├── movies.csv          
│   ├── wordcloud_text.csv  
│   ├── movie_similarity_network.csv  
│   └── user_recommendations/  
│       ├── user_1_recommendations.csv
│       └── user_2_recommendations.csv
├── models/                 
├── src/                    
├── web/                   # Web 
│   ├── app.py             # Flask 
│   └── templates/          
│       └── dashboard.html  
└── main.py                

## Run
launch:
`bashpython main.py --mode all`

only generate recommend:
`bashpython main.py --mode recommend`

Only start the Web service:
bashpython main.py --mode serve

Generate recommendations for specific users:
`bashpython main.py --mode recommend --user_id 1`


## --DVC workflow--

dvc for big data sync. saved in google drive.

1. init & setting

```
git clone git@github.com:kkk8670/flick-pick.git

cd <project>
dvc init

# setting remote path
dvc remote add -d gdrive gdrive://1tWIBZaOl4ASztgRRYnE8HfCZFmGbGuFS   # folder data id
dvc remote modify gdrive --local gdrive_use_service_account true
dvc remote modify gdrive --local gdrive_service_account_json_file_path .secrets/dvc-service-account.json

# get data
dvc pull -r gdrive
ls data
```

2. workflow

graph LR
    A[file changed] --> B["git commit"]
    B --> C["git push"]
    C --> D[auto run dvc push]
```
# ** notice ** pls pull code & data before work
git pull 
dvc pull -r gdrive

# work with code & data alteration

# submit if data changed
dvc add data
git add data.dvc .gitignore
git commit -m "xxx"
dvc push -r gdrive
git push  # push git will auto trigger dvc push
```


## Other Issue

### install issue

- If there is error `No module named 'distutils'`，when python >= v3.12, 
run 
	`pip install -U pip setuptools wheel`
or 
	`pip install packaging`

- local Spark dependents on `openjdk@17`

### exception handling

- If data is not synchronized after git push
run
	`dvc push -r gdrive`
Check hook
	`ls -la .git/hooks/post-commit`
