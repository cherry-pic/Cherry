# Cherry Dataset

By: [Author1](TODO:url) `<TODO:email>` and [Author2](TODO:url) `<TODO:email>`

This dataset was created as apart of a study on identifying censored statements in news reporting, a type of bias known in literture as "cherry-picking" of facts. The dataset is annotated for statement importance with regard to a specific event. We call this dataset the **Cherry**; what follows below is the datasheet describing this data. If you use this dataset, please acknowledge it by citing the original paper:

```
@inproceedings{cao2019toward,
  title={Filling the Blanks: Context-driven Detection and Correction of Cherry-picking in
News Reporting},
  author={},
  booktitle={},
  year={2025},
  note={}
}
```


## Motivation


1. **For what purpose was the dataset created?** *(Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.)*
    
    This dataset was created as part of a study to identify and correct cherry-picking of statements in news reporting based on their importance for an event. The dataset provides annotations on the importance of statements with regard to an event covered by different news sources.


1. **Who created this dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?**
    
    This info is currently withheld for anonymity purposes.


1. **Who funded the creation of the dataset?** *(If there is an associated grant, please provide the name of the grantor and the grant name and number.)*
    
    This info is currently withheld for anonymity purposes.


1. **Any other comments?**
    
    None.





## Composition


1. **What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?** *(Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.)*
    All the instances of this dataset are textual data. The instances represent statements (sentences) taken from news reports from different sources covering a certain event, and a context about the event.
    

2. **How many instances are there in total (of each type, if appropriate)?**
    
    The dataset consists of 3,346  intances (i.e., triplets), the distribution of the instances per class is indicated in the table below. This distribution depends on the configuration where we aggregate the different levels of importance under a single or multiple labels.
     * The first configuration is binary where only very important statements are under class 1 and any other importance level **including** few incorrect excerpts and statements that annotators were not sure about their importance are under class 2.
     * The second configuration is binary where only very important statements are under class 1 and any other importance level **excluding** few incorrect excerpts and statements that annotators were not sure about their importance are are under class 2.
     * The third configuration is multi-class where very important statements are in class 1, other importance levels are class 2, and incorrect excerpts or statements annotators were not sure about are class 3.
     * The fourth configuration is multi-class where very important statements are in class 1, kind of importance level is class 2, and not very important statements are under class 3.

    The distribution of labels on classes for each of the aformeentioned four configurations is:

    | Configuration | Class 1 |  Class 2   | Class 3  | 
    |:--------------|:-------:|:----------:|:--------:| 
    | 1             |    2175 (64%)     |   1232 (36%)   |    -     |  
    | 2             |   2175 (65%)    |   171 (35%)    |    -     |  
    | 3             |   2175 (64%)     |   1171 (34%)   |   61 (2%)   |  
    | 4             | 2175 (65%)  | 667 (20%) | 504 (15%) | 

    

3. **Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?** *(If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).)*
    
    It is a sample taken from a collection of news events. The most controversial events from the larger set of events are selected for the sample. The need for events to be controversial ensures a better coverage from different news sources for the purposes of this study.


4. **What data does each instance consist of?** *(``Raw'' data (e.g., unprocessed text or images)or features? In either case, please provide a description.)*
    
    Each instance is a triplet of:
   1. An English statement (sentece) taken from a news report that covered a certain event.
   2. An English news report that gives some context about the event, collected from a neutral source, or in some cases sources with counter biases (w.r.t. the source of the statement)
   3. A numerical label (0,1, or 1-3 in some cases) that indicates if the statement is important with regard to the event or not or the level of importance.
`

5. **Is there a label or target associated with each instance? If so, please provide a description.**
    
    The labels are numerical values indicating the importance of the stattement with regard to the event described in the context.


6. **Is any information missing from individual instances?** *(If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.)*
    
    No.


7. **Are relationships between individual instances made explicit (e.g., users' movie ratings, social network links)?** *( If so, please describe how these relationships are made explicit.)*
    
    Instances are related (i.e., taken from news reports that cover the same event) if their context is similar. However, for this specific study, this relationship is not important, and thus not explicitly indicated.

8. **Are there recommended data splits (e.g., training, development/validation, testing)?** *(If so, please provide a description of these splits, explaining the rationale behind them.)*
    
    The recommended split ratio is 85% for training and 15% for testing. An important thing to consider to avoid any data leakeage when training supervised models on this dataset is to split on the event level. This means that stataements with the same context should be places within the same split.

9. **Are there any errors, sources of noise, or redundancies in the dataset?** *(If so, please provide a description.)*
    
    There are negligible errors resulting from tokenization on the sentence level (e.g., the existence of extra quotes or missing full stops). Additionally, if a statement is incorrect it is indicated with its label.


10. **Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?** *(If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.)*
    
    The dataset is self-contained.


11. **Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals' non-public communications)?** *(If so, please provide a description.)*
    
    No; all documents in the data are published news reports.


12. **Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?** *(If so, please describe why.)*
    
    No; all documents in the data are published news reports.


13. **Does the dataset relate to people?** *(If not, you may skip the remaining questions in this section.)*
    
    No.


14. **Does the dataset identify any subpopulations (e.g., by age, gender)?** *(If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.)*
    
    N/A.


15. **Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?** *(If so, please describe how.)*
    
    N/A.


16. **Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?** *(If so, please provide a description.)*
    
    N/A.


17. **Any other comments?**
    
    None.





## Collection Process


1. **How was the data associated with each instance acquired?** *(Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.)*
    
    The dataset was collected from published news reports. Each instance is a sentence from these news reports and a full report.


2. **What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?** *(How were these mechanisms or procedures validated?)*
    
    News reports were collected using The GDELT Project's events database API [GDELT](https://www.gdeltproject.org/).


3. **If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?**
    
    The sample collected was conditioned on the date of publication (news reports published between December 2019 and January 2021) and selected news sources. The list of the sources is available in this [sheet](https://github.com/cherry-pic/Cherry/blob/master/news_outlets.txt).


4. **Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?**
    
    All collection and annotation was done by the authors, in addition to Ph.D. students.


5. **Over what timeframe was the data collected?** *(Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)?  If not, please describe the timeframe in which the data associated with the instances was created.)*
    
    December 2019 - January 2021.


6. **Were any ethical review processes conducted (e.g., by an institutional review board)?** *(If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.)*
    
    No review processes were conducted with respect to the collection and annotation of this data as these are published news reports.


7. **Does the dataset relate to people?** *(If not, you may skip the remaining questions in this section.)*
    
    No.


8. **Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?**
    
    N/A.


9. **Were the individuals in question notified about the data collection?** *(If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.)*
    
    N/A.


10. **Did the individuals in question consent to the collection and use of their data?** *(If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.)*
    
    N/A.


11. **If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?** *(If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).)*
    
    N/A.


12. **Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?** *(If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.)*
    
    N/A. 


13. **Any other comments?**
    
    None.





## Preprocessing/cleaning/labeling


1. **Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?** *(If so, please provide a description. If not, you may skip the remainder of the questions in this section.)*
    
    Only the version that is used by the supervised models was preprocessed by removing stop words. 


2. **Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?** *(If so, please provide a link or other access point to the "raw" data.)*
    
    Yes, all the different versions of the data are available.


3. **Is the software used to preprocess/clean/label the instances available?** *(If so, please provide a link or other access point.)*
    
    N/A.


4. **Any other comments?**
    
    None.





## Uses


1. **Has the dataset been used for any tasks already?** *(If so, please provide a description.)*
    
    The dataset has been used to identify important statements in a given news report in order to detect and correct cherry-picking of statements.


2. **Is there a repository that links to any or all papers or systems that use the dataset?** *(If so, please provide a link or other access point.)*
    
    No.


3. **What (other) tasks could the dataset be used for?**
    
    The dataset could possibly be used for other tasks related to statement importance within an article, such as summarization. 


4. **Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?** *(For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other undesirable harms (e.g., financial harms, legal risks)  If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?)*
    
    No.


5. **Are there tasks for which the dataset should not be used?** *(If so, please provide a description.)*
    
    No.


6. **Any other comments?**
    
    None.




## Distribution


1. **Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?** *(If so, please provide a description.)*
    
    Yes, the dataset is freely available.


2. **How will the dataset will be distributed (e.g., tarball  on website, API, GitHub)?** *(Does the dataset have a digital object identifier (DOI)?)*
    
    The dataset is free for download at https://github.com/cherry-pic/Cherry.


3. **When will the dataset be distributed?**
    
    The dataset is distributed as of Jan 2024.


4. **Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?** *(If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.)*
    
    The dataset is licensed under the GNU General Public License v3.0.


5. **Have any third parties imposed IP-based or other restrictions on the data associated with the instances?** *(If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.)*
    
    Not to our knowledge.


6. **Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?** *(If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.)*
    
    Not to our knowledge.


7. **Any other comments?**
    
    None.





## Maintenance


1. **Who is supporting/hosting/maintaining the dataset?**
    
    Authors.


2. **How can the owner/curator/manager of the dataset be contacted (e.g., email address)?**
    
    E-mail addresses withheld temprarily for anonymity purposes and will be provided leter.


3. **Is there an erratum?** *(If so, please provide a link or other access point.)*
    
    Currently, no. As errors are encountered, future versions of the dataset may be released (but will be versioned). They will all be provided in the same github location.


4. **Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances')?** *(If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?)*
    
    Same as previous.


5. **If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?** *(If so, please describe these limits and explain how they will be enforced.)*
    
    N/A.


6. **Will older versions of the dataset continue to be supported/hosted/maintained?** *(If so, please describe how. If not, please describe how its obsolescence will be communicated to users.)*
    
    Yes; all data will be versioned.


7. **If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?** *(If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.)*
    
    Errors may be submitted via the bugtracker on github. More extensive augmentations may be accepted at the authors' discretion.


8. **Any other comments?**
    
    None.


