import codecs
from helpers import set_seed
from data_loading import prepare_data
from data_cleaning import clean_data
from train import train_model
import torch
from datetime import datetime
import os
os.environ['TRANSFORMERS_CACHE'] = '/work/09875/israajaradat/ls6/.cache/huggingface/'
now = datetime.now()
now = now.strftime("%d_%m_%Y_%H_%M")

###################################################################################################
# P A R A M E T E R S
###################################################################################################
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("This experiment will run on "+str(DEVICE))
PRETRAINED_MODEL = "allenai/longformer-base-4096"  # 'bert-large-uncased','albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2', 'bert-base-uncased', ...
FREEZE_PRETRAINED = True  # if True, freeze the encoder weights and only update the classification layer weights
BATCH_SIZE = 8  # batch size
LR = 2e-05  # learning rate
EPOCHS = 5  # number of training epochs
DROPOUT = 0.0 # droput at the classification dense layer
HIDDEN_SIZE = 768 if 'base' in PRETRAINED_MODEL else 1024 # size of the vector representations of each token: 768 for base models, 1024 for large models
PREDICTION_THRESHOLD = 0.5
GLOBAL_ATTENTION_LOCATION = [0,1] # sets the location of Longformer's global attention ([0]:classification token [CLS], [1]: first sequence (fact) only, [2]: 2nd sequence (context) only, [0,1]: CLS+fact, [0,1,2] all sequence, etc..
CLASSIFICATION_TYPE = 1 # choose from 1-5 classification types (refer to the Cherry Google doc)
REMOVE_REDUNDANT = True
SUMM_CONTEXT = True   # use summarized biased articles as context
FIXED_CONTEXT = False  # use fixed context length measured by a fixed number of words.
RAW_DS_PATH = "data/classification_" + str(CLASSIFICATION_TYPE) + "/raw_ds_classif_type_" + str(CLASSIFICATION_TYPE) + '.csv'
MAX_SEQ_LENGTH = 512  # the max length of the input sequence to BERT
VALIDATION="holdout" #"cv" data is split into test and train for cross validation, "holdout": data is split into train, dev, test and holdout evaluation occurs
TEST_SIZE = 0.15
SHUFFLE = False
PADDING_STRATEGY = "max_length" # choose from 'max_length' , False/'do_not_pad', or 'longest'
CV_K = 5 # number of folds in cross validation. (choose a number between 5-full data set size) because the size of the test split changes with CV_K, gets bigger when K decreases
set_seed(1) #  Set all seeds to make reproducible results
EXP_DIR = "outputs/Exp_"+now+"_LR_"+str(LR)+"_seq_"+str(MAX_SEQ_LENGTH)+"/"
message = input("Experiment highlight: ")
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)
with codecs.open(EXP_DIR+"config",'w',encoding='utf8') as log:
    log.write("Device:"+str(DEVICE)+'\nPretrained_model:'+PRETRAINED_MODEL+'\nFreeze_pretrained:'+str(FREEZE_PRETRAINED)+'\nBatch_size:'+str(BATCH_SIZE)+'\nLearning_rate:'+str(LR)+'\nEpochs:'+str(EPOCHS)+'\nDroput:'+str(DROPOUT)+'\nPrediction_threshold:'+str(PREDICTION_THRESHOLD)+'\nMax_sequence_length:'+str(MAX_SEQ_LENGTH)+'\nPadding_strategy:'+str(PADDING_STRATEGY)+'\nValidation_type:'+VALIDATION+'\nExperiment highlights: '+message)
    log.close()
print("The outputs and artifacts of this experiment will be saved in directory: "+EXP_DIR)

###################################################################################################
# D A T A    C L E A N I N G    &    L O A D I N G
###################################################################################################
data_splits = clean_data(raw_ds_path=RAW_DS_PATH,validation_type=VALIDATION,cv_k=CV_K,remove_redundants=REMOVE_REDUNDANT,
                         classification_type=CLASSIFICATION_TYPE,shuffle = SHUFFLE,test_size=TEST_SIZE,summ_context_data=SUMM_CONTEXT,fixed_context=FIXED_CONTEXT)
dataloaders,test_dfs = prepare_data(data_splits, BATCH_SIZE, PRETRAINED_MODEL,MAX_SEQ_LENGTH,GLOBAL_ATTENTION_LOCATION,PADDING_STRATEGY)

###################################################################################################
# T R A I N I N G   &   V A L I D A T I O N   &   T E S T I N G
###################################################################################################
train_model(LR, dataloaders, EPOCHS, DEVICE, PRETRAINED_MODEL,FREEZE_PRETRAINED,DROPOUT,HIDDEN_SIZE,PREDICTION_THRESHOLD,test_dfs,EXP_DIR,CLASSIFICATION_TYPE)

