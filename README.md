# NLP-Final
NLP final project

The project write up is the file named "nlp_final.pdf"

video presentation: https://www.youtube.com/watch?time_continue=3&v=6o87eiryc6s&feature=emb_logo

Running Project
Steps:
- create a python virtual environment
- install all of the requirements with ```pip install -r requirements.txt``` (sorry, alot of the packages are from old projects)
- download this: http://nlp.stanford.edu/data/glove.6B.zip (file too big to put in repo, need it for vector embedding)
- save glove.6B.100d.txt in a folder called 'glove' in the same level as test.py
- create an empty folder for computer generated summaries to go, put in the same level as test.py
- run the program with:
- ```python test.py <shelter in place .txt folder name> <computer generated summaries folder> <human summary folder>```
Outer Directory should look like this:
```
│   requirements.txt
│   test.py
│
├───cg_summaries (empty dir, will get filled by computer summaries)
│
├───glove
│       glove.6B.100d.txt
│       glove.6B.200d.txt
│       glove.6B.300d.txt
│       glove.6B.50d.txt
│
├───human_summaries
│       Alabama.txt
│       Alaska.txt
│       Athens-Clarke.txt
│       California.txt
│       Delaware.txt
│       Douglas.txt
│       Hawaii.txt
│       Jackson.txt
│       Johnson.txt
│       Leavenworth.txt
│       Maine.txt
│       Maryland.txt
│       Montana.txt
│       New Hamshire.txt
│       New Jersey.txt
│       New Mexico.txt
│       Orange County FL.txt
│       Pennsylvania.txt
│       Salt Lake.txt
│       Savannah.txt
│       Tennessee.txt
│       Vermont.txt
│       Washington.txt
│       Wisconsin.txt
│
└───sips
        Alabama.txt
        Alaska.txt
        Athens-Clarke.txt
        California.txt
        Delaware.txt
        Douglas.txt
        Hawaii.txt
        Jackson.txt
        Johnson.txt
        Leavenworth.txt
        Maine.txt
        Maryland.txt
        Montana.txt
        New Hamshire.txt
        New Jersey.txt
        New Mexico.txt
        Orange County FL.txt
        Pennsylvania.txt
        Salt Lake.txt
        Savannah.txt
        Tennessee.txt
        Vermont.txt
        Washington.txt
        Wisconsin.txt
```
