<!-- ABOUT THE PROJECT -->
## Feature Description
<p>This feature generates recommended upper bound and lower bound guidance time to be set for a question based on the attributes of the questions provided.</p>
It is driven by a RandomForest model, and returns the bounds in bucket size of 16 minutes.
The function and its dependencies are hosted as a docker image on ECR and is deployed on AWS lambda. 

## Usage
The input to the function is in this JSON format:
```
{
  "questionInfo": {
    "questionnumber": 5,
    "modulename": "module name",
    "setnumber": 6,
    "questiontitle": "title",
    "masterContent": "content",
    "partPosition": 3,
    "workedsolution": "solContent",
    "partContent": "partContent",
    "skill": 1
  }
}
```

The function returns data in the following format:
```
'statusCode': 200,
'body': json.dumps({"upperBound" : upperBound,
                    "lowerBound": lowerBound})
```


### Dependencies

<ul>
  <li>spacy==3.7.4</li>
  <li>pickleshare==0.7.5</li>
  <li>nltk==3.8.1</li>
  <li>numpy==1.26.1</li>
  <li>sscikit-learn==1.4.0</li>
</ul>








