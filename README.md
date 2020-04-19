# NLP-Final
NLP final project

Steps:
- create a python virtual environment
- install all of the requirements with "pip install -r requirements.txt" (sorry, alot of the packages are unneccessary from old projects)
- download this: http://nlp.stanford.edu/data/glove.6B.zip (file too big to put in repo, need it for vector embedding)
- save glove.6B.100d.txt in a folder called 'glove' in the same level as test.py
- create an empty folder for computer generated summaries to go, put in the same level as test.py
- run the program with: "py test.py (shelter in place .txt folder name) (computer generated summaries folder) (human summary folder)"

Outer Directory should look like this:
├───cg_summaries
├───glove| 
         -- glove.6B.100d.txt
├───human_summaries
├───pdfs to convert
└───Shelter In Place Orders
|--- test.py
|--- test2.py
