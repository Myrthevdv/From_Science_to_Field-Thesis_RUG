# From Science to Field

## Leveraging an Open-Source Language Model for Tailored Summaries of Sport Science Articles for Athletes

### Bachelor Thesis ~ BSc Information Science at the University of Groningen
### Myrthe van der Veen 
### S4977068

### The Repository
This repository consist of a the code and data used for the thesis from science to field: Leveraging an Open-Source Language Model for Tailored Summaries of Sport Science Articles for Athletes

### Habrok


Habrok is only usable for people from the RUG. 
Before you can log in to Habrok, you need to get access using the following link: https://wiki.hpc.rug.nl/habrok/introduction/policies.

When you have access can log in to Habrok and use the open-source LLM Phi3 with the following prompts:

```
# Login:
ssh -Y s-nummer@login1.hb.hpc.rug.n

# To submit the job/code:
sbatch phi3.sh

# Show a diagram with the information about running. NUMBER is the number that is shown when you submit the job.
Jobinfo JOBNUMBER

# Shows the running results of the model for a specific job “NUMBER”
nano slurm-NUMBER.ou
``` 

#### Data folder
- ```Annotated data.xlsx```
    - This dataset contains 15 selected articles about "sport injuries" from PMC, including their Abstract, Introduction, and Conclusion (AIC). Additionally, for each article, the dataset includes three summaries generated by three different closed-source LLMs (ChatGPT, Cohere, and Gemini). The dataset also includes annotations for each summary, which consist of the overall relative rank (1-3), the rankings (1-5) on different evaluation criteria: Clarity and Accessibility, Inclusion of Key Findings, and Relevance for Athletes and each ranking comes with explanations of the choices made for the ranking on these three evaluation criteria for each summary.
- ```Development_set.xlsx```
    - This dataset contains the last 4 articles, including rankings and explenations from the ```Annotated data.xlsx``` which is used in the development phase. 
- ``` Development_set_summaries_and_BERTScores.xlsx ```
    - This dataset contains for the first 3 summaries of the  ```Development_set.xlsx```, 3 different summaries in which these 3 summaries are used in the few-shot prompt to create a summary of the last article in the ```Development_set.xlsx```. Together with the Bertscore which is obtained by comparing these summaries with the best-annotated overall relative ranked summary from the ```Development_set.xlsx``` for this last article in the ```Development_set.xlsx```. 
- ```Evaluation_set.xlsx```
    - This dataset contains the first 11 articles, including rankings and explenations from the ```Annotated data.xlsx``` which is used in the evaluation phase. 
- ```Summarized_Articles_Phi3.xlsx```
    - This dataset contains the 11 articles from the ```Evaluation_set.xlsx``` that are summarized by the open-source LLM Phi3 by the 4 different prompts
 
#### Data/evaluation_scores folder

- ```BERTScores_summarized_articles.xlsx```
    - This dataset contains the BERTScore of all the summaries generated with the open-source LLM Phi3 from the ```Evaluation_set.xlsx```, compared with the annotated relative (rank1,rank2,rank3) summaries generated by the closed-source LLM
- ```Readability_scores_summarized_articles.xlsx```
    - This dataset contains the Flesch Reading Ease Scores and Flesch Kincaid Grade Levels of all the summaries generated with the open-source LLM Phi3 from the ```Evaluation_set.xlsx```
