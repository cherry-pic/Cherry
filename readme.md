# Cherry: On Detecting Cherry-picking in News Coverage Using Large Language Models
## Abstract: 
Cherry-picking involves suppressing (censoring) or distorting evidence that supports the counter argument. Cherry-picking facts in news reports by mainstream media distorts public perception, undermines trust, and fuels misinformation by presenting a biased or incomplete narrative. Manually identifying suppressed statements in news stories is challenging and time consuming. In this study, we introduce a novel importance-based approach to automatically spotting and correcting cherry-picking by identifying then substituting missing important statements in a target news story with the help of contextual information from other news sources with different biases. Additionally, we showcase the flexibility of our approach by utilizing different methods to estimate a statement's importance including fine-tuned embedding models, zero and few-shot generative models, in addition to unsupervised methods. Furthermore, this research introduces a novel dataset specifically designed for training and evaluating cherry-picking detection methods. Our best performing method achieves an F-1 score of about 89% in estimating a statement's importance. Moreover, results show the effectiveness of the proposed approach in correcting cherry-picking by mitigating cherry-picking and bringing the biased narrative closer to its neutral alternative by nearly 12%. Finally, through thorough experimentation, we provide answers to a set of important research questions related to cherry-picking detection and correction.
## Contents:
This repo contains three main directories:
1. cherry_baseline_CUDA: contains the BERT variant of the model.
2. cherry_baseline_CUDA_nocontext: contains the BERT variant of the model that accepts only the statement, without context in th einput sequence.
3. cherry_longformer_CUDA: contains the Longformer variant of the model.
4. cherry_picking_detection: contains the scripts required to run the full-end-to-end cherry-picking detection pipeline.
5. generative_LLM: contains the scripts and data required to run GPT experiments.
6. lexrank: contains the scripts and data required to run LexRank experiments.
7. Results: contains the expeirment results and visualizations.
8. experiments: contains the predictions and all experiment details and results grouped by model.
9. cherry-picking_correction: contains the scripts and data files required to run the full-end-to-end correction of cherry-picking and measure the improvement in each article's biases by comparing them and the original versions to the neutral narrative.
## Install and run:
### Training the supervised models
1. Use the "environment.yml" file to recreate the environment. </br>
2. To run traning for any of the supervised model variants above, run the following command from inside the variant's main directory: </br>
```python main.py ```</br>
3. To modify the paramters, adjust the values in the main.py files under "paramters".
Each of the variants directories contians the code and the data sets used in the four different classification configurations. You do not need to reset data paths, just choose the classification configuration number in the paramter list in main.py.

### End-to-end cherry-picking detection:
To run the end-to-end detection pipleine using the top-performing model, install the large files from [here](https://drive.google.com/drive/folders/1bJTSS5HJdb2GGEmfnOciIHnn9U6qOFg4?usp=sharing) and place them under the cherry_picking_detection directory, then run the following command in the directory: </br>
```python spot_cherry_picking.py``` </br>
The script will run the pipline on the clustered and prepared data file "bias_analysis_events_clustered_wpredictions.json"
This file contains all the inference data comprised of the 2453 unseen events preprocessed, clustered, and also conatains the results of inference from the top-performing model.

### Prediction using unsupervised / few-shot learners models:
To run these models to predict on the testing data set, run either "gpt_exp.py" or "lexRank_exp.py".

### End-to-end cherry-picking correction:
To run the end-to-end correction pipleine use the data file with model's predictions from the detection step. For instance, you can use the file "bias_analysis_events_clustered_wpredictions.json"  [here](https://drive.google.com/drive/folders/1bJTSS5HJdb2GGEmfnOciIHnn9U6qOFg4?usp=sharing) and place it under the cherry_picking_correction directory, then run the following command in the directory: </br>
```python correct_cherry_picking.py``` </br>
