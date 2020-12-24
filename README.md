# Final Project
Final project for class ML Security for Cyber

## Validation Data
Download BadNet models and the clean validation dateset and store them under models/ and data/ directory respectively.  
Download link: https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab

## Evaluate
1. Generate a “repaired” model, execute repair.py by running:  
python repair.py \<badnet model directory\> \<clean validation data directory\>  
“Repaired” model G will be saved as models/\<badnet model name\>_defence.h5  
E.g.,  
python repair.py models/anonymous_1_bd_net.h5 data/clean_validation_data.h5  
Saved model: models/anonymous_1_bd_net_defence.h5.  
Note: if you use existed models/\<badnet model name\>_defence.h5 files, you can skip this step.  


2. Evaluate, execute eval_defence.py by running:  
python eval_defence.py \<image directory\> \<badnet model name\>  
It will print the class number in [1, N+1].  
E.g.,  
python eval_defence.py img/img_9.png anonymous_1_bd_net  
Output: 855  

