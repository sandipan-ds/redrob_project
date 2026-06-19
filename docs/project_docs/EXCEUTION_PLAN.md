1. The refence for shortlisting the resumes is based on the two docs- 
 a. docs\reference_docs\job_description.docx
 b. docs\reference_docs\redrob_signals_doc.docx (it is secondary signal provider for ranking)
 we will condider both but objectiverly

2. Rather than assuming the criterias, and weightage of criterias we should take input form the recruiter
how much each critreia does the recruiter values based on an input criteria scre. Even before we do that we should first map the 
docs\reference_docs\redrob_signals_doc.docx with docs\reference_docs\job_description.docx and examples from [text](../../data/sample_candidates.json) (see the fields of thsoe candiadte profiles)
how many of the critreia in these three docs are common? are there additional criterias from both of these docs apart from the mapped common criteria for shortlisting (for exampple the summary field) ? how much of teh jD is relevant for selcting the json based candidate profiles, and how much of the JD is unncessary, we must figure that out. Once we map te cos and take the relevant critreia from both the docs that are signals and not noise for the candidate selction. Then we move to the next step. (probable solution a good prompt toforce the LLM to get the signal or useful criterais based on the JD and the redrob_signals_doc)

3. Take the wieghtage score form the recruiter for example-

Python : 8/10
SQL : 6/10
LangChain : 8/10
Eye-to-detail: 8/10
Education Graduation : 8/10 (Here the score is based on Tier of teh college Tier 1 is best)
Experince : 8/10
Role Data Scientist : 8/10 (But here is catch what if teh candidate has worekd as software negineer, data analyst, busness nalyst, data engineer etc, would they get the same score?)
Endorsements (a linear scale, min endorsemnets a candiadte should have) : 8/10 (that means 8 is at most for min number of endorse menst say 30, 0 is least anything below 30 will be linearly calcukated take a scale of 0 to 30 endorsements)
CGPA : 10 is atmost but how imprtant is this ? ask for min CGPA say its 7 , so 7 becomes the baseline 
    so suppose teh recruiter rates it as 8/10 (10 will correspond to the score of 8, and 7 at 1, 0 to 7 is on teh scale of 1 so its not alinera scale as the min CGPA required is 7.)

These are just examples.

4. Once you get all these wightage critreias. This documnet should now serve as the ground truth for evaluating each candidate profie.

5. Now you may base LLM based ereasoning on this ground truth documnet, and the candidate profile json. (A prompt on this shoudl be exclusively stated on howth reasoning shoudl wokr out, ishouldnt be too long but detailed, I think 100 words reasoning is fine.)

6. The project has latency as critria for that reda teh doc-
docs\reference_docs\submission_spec.docx

7. There are also Honeypots so be careful about that, I think total exp of the candidates if it doesnt match with indivdual organisation exp it can be a hineypot, teher are more examples in the - docs\reference_docs\submission_spec.docx, and it is a critreia for rejection.

