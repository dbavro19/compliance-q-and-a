import boto3
import botocore
import json
from opensearchpy import OpenSearch
from opensearchpy import RequestsHttpConnection, OpenSearch, AWSV4SignerAuth
import streamlit as st





config = botocore.config.Config(connect_timeout=300, read_timeout=300)
bedrock = boto3.client('bedrock-runtime' , 'us-east-1', config = config)

#Setup Opensearch connectionand clinet
host = 'HOSTNAME' #use Opensearch Serverless host here
region = 'REGION'# set region of you Opensearch severless collection
service = 'aoss'
credentials = boto3.Session().get_credentials() #Use enviroment credentials
auth = AWSV4SignerAuth(credentials, region, service) 

oss_client = OpenSearch(
    hosts = [{'host': host, 'port': 443}],
    http_auth = auth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection,
    pool_maxsize = 20
)

#
def get_embeddings(bedrock, text):
    body_text = json.dumps({"inputText": text})
    modelId = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    contentType='application/json'

    response = bedrock.invoke_model(body=body_text, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')

    return embedding

#Get KNN Results
def get_knn_results(client, userVectors):

    query = {
        "size": 5,
        "query": {
            "knn": {
                "vectors": {
                    "vector": userVectors, 
                    "k": 5
                }
            }
        },
        "_source": False,
        "fields": ["content", "source", "page"],
    }


    response = client.search(
        body=query,
        index='INDEXNAME',
    )



    similaritysearchResponse = ""
    count = 0
    for i in response["hits"]["hits"]:
        content = "Page Content: " + str(i["fields"]["content"])
        source = "Source: " + str(i["fields"]["source"])
        page_number = "Page Number: " + str(i["fields"]["page"])
        new_line = "\n"

        similaritysearchResponse =  similaritysearchResponse + content + new_line + source + new_line + page_number + new_line + new_line
        count = count + 1
    
    #print("----------------------Similarity Search Results-----------------------")
    #print(similaritysearchResponse)
    #print("---------------------END Similarity Search Results--------------------")

    return similaritysearchResponse


def parse_xml(xml, tag):
  start_tag = f"<{tag}>"
  end_tag = f"</{tag}>"
  
  start_index = xml.find(start_tag)
  if start_index == -1:
    return ""

  end_index = xml.find(end_tag)
  if end_index == -1:
    return ""

  value = xml[start_index+len(start_tag):end_index]
  return value


def get_knn_keyword_results(client, userVectors):

    query = {
        "size": 3,
        "query": {
            "knn": {
                "vectors": {
                    "vector": userVectors, 
                    "k": 3
                }
            }
        },
        "_source": False,
        "fields": ["content", "source", "page"],
    }


    response = client.search(
        body=query,
        index='INDEXNAME',
    )



    keyword_similarity_search_response = ""
    count = 0
    for i in response["hits"]["hits"]:
        content = "Page Content: " + str(i["fields"]["content"])
        source = "Source: " + str(i["fields"]["source"])
        page_number = "Page Number: " + str(i["fields"]["page"])
        new_line = "\n"

        keyword_similarity_search_response =  keyword_similarity_search_response + content + new_line + source + new_line + page_number + new_line + new_line
        count = count + 1
    
    #print("----------------------Similarity Search Results-----------------------")
    #print(keyword_similarity_search_response)
    #print("---------------------END Similarity Search Results--------------------")

    return keyword_similarity_search_response


def invoke_llm(bedrock, user_input, knn_results, keyword_results, keyword_knn_results):

# Uses the Bedrock Client, the user input, and the document template as part of the prompt


    ##Setup Prompt
    system_prompt = f"""

You are a financial compliance expert who will assist users in understanding their compliance and regulatory requirements for Financial operations in Europe
Based on the provided context in <semantic_search_results>, <keyword_search_results>, and <keyword_semantic_search_results>, Answer the user's question in a concise manner, provide sources the source link annotation inline and refer to the source page number to where the relevant information can be found. Provide refrences to the source of the infomration in line, and the annotation at the end your response
Do not include any information outside of the information provided in <semantic_search_results> and <keyword_search_results>. If the provided context does not contain a valid answer to the question, say so
Weight the information in <semantic_search_results> slightly higher than the information in <keyword_search_results> and <keyword_semantic_search_results>.
Your answer should be straight forward and simple, dont use examples or make overly specific references to examples or fields
Do not include any preamble


<semantic_search_results>
{knn_results}
</semantic_search_results>

<keyword_search_results>
{keyword_results}
</keyword_search_results>

<keyword_semantic_search_results>
{keyword_knn_results}
</keyword_semantic_search_results>

Make sure your response is well formatted and human readable 

"""

    prompt = {
        "anthropic_version":"bedrock-2023-05-31",
        "max_tokens":1000,
        "temperature":0.5,
        "system" : system_prompt,
        "messages":[
            {
                "role":"user",
                "content":[
                {
                    "type":"text",
                    "text": "<user_question>" +user_input +"</user_question>"
                }
                ]
            },
            {
                    "role": "assistant",
                    "content": "Based on the provided context:"
            }
        ]
    }

    json_prompt = json.dumps(prompt)

    response = bedrock.invoke_model(body=json_prompt, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")

    #modelId = "anthropic.claude-v2"  # change this to use a different version from the model provider if you want to switch 
    #accept = "application/json"
    #contentType = "application/json"
    #Call the Bedrock API
    #response = bedrock.invoke_model(
    #    body=body, modelId=modelId, accept=accept, contentType=contentType
    #)

    #Parse the Response
    response_body = json.loads(response.get('body').read())

    llmOutput=response_body['content'][0]['text']

    #Return the LLM response
    return llmOutput



def extract_keywords(bedrock, user_input):

# Uses the Bedrock Client, the user input, and the document template as part of the prompt


    ##Setup Prompt
    system_prompt = f"""

You are a financial compliance expert who will assist users in understanding their compliance and regulatory requirements for Financial operations in Europe
Based on the provided question from the user in <user_question>, extract the top 3 keywords that would be the most helpful for a search
Use the details in tips and tricks as a reference on how to generate your key words

<tips_and_tricks>
The first keyword should be a single word that capture the most significant search context (ie. questions about error codes should have the first keyword be the error code in question, and ONLY the error code)
The second keyword should be a concise (2-3 word) search term that would capture the core search question (ie. NCA rejection)
the 3rd keyword should resemble a concise google search for the main semantic context of the question. And should be about 4 to 5 words
</tips_and_tricks>


Return your top 3 key word searches in <keyword_1>, <keyword_2>, and <keyword_3>. Dont include any other text other than the keywords in your response
"""

    prompt = {
        "anthropic_version":"bedrock-2023-05-31",
        "max_tokens":1000,
        "temperature":0.5,
        "system" : system_prompt,
        "messages":[
            {
                "role":"user",
                "content":[
                {
                    "type":"text",
                    "text": "<user_question>" +user_input +"</user_question>"
                }
                ]
            },
            {
                    "role": "assistant",
                    "content": "Based on the provided context:"
            }
        ]
    }

    json_prompt = json.dumps(prompt)

    response = bedrock.invoke_model(body=json_prompt, modelId="anthropic.claude-3-haiku-20240307-v1:0", accept="application/json", contentType="application/json")

    #modelId = "anthropic.claude-v2"  # change this to use a different version from the model provider if you want to switch 
    #accept = "application/json"
    #contentType = "application/json"
    #Call the Bedrock API
    #response = bedrock.invoke_model(
    #    body=body, modelId=modelId, accept=accept, contentType=contentType
    #)

    #Parse the Response
    response_body = json.loads(response.get('body').read())

    llmOutput=response_body['content'][0]['text']

    keyword_1 = parse_xml(llmOutput, "keyword_1")
    keyword_2 = parse_xml(llmOutput, "keyword_2")
    keyword_3 = parse_xml(llmOutput, "keyword_3")

    print(f"keyword_1: {keyword_1}")
    print(f"keyword_2: {keyword_2}")
    print(f"keyword_3: {keyword_3}")

    #Return the LLM response
    return keyword_1, keyword_2, keyword_3


def get_keyword_results(client, keyword):

    query_1={
        "size": 3,  # Limits the number of search results to 3
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "content": keyword
                        }
                    }
                ],
                "minimum_should_match": 1  # Ensures at least one 'match' must be true
            }
        },
        "_source": ["content", "source", "page"],  # Specify which fields to include in the response
        "highlight": {  #Optional, to highlight matches in the content field
            "fields": {
                "content": {}
            }
        }
    }


    response_1 = client.search(
        body=query_1,
        index='INDEXNAME',
    )



    keyword_results = ""
    count = 0
    for i in response_1["hits"]["hits"]:
        content = "Page Content: " + str(i["_source"]["content"])
        source = "Source: " + str(i["_source"]["source"])
        page_number = "Page Number: " + str(i["_source"]["page"])
        new_line = "\n"

        keyword_results =  keyword_results + content + new_line + source + new_line + page_number + new_line + new_line
        count = count + 1
    
    #print("----------------------Similarity Search Results-----------------------")
    #print(keyword_results)
    #print("---------------------END Similarity Search Results--------------------")

    return keyword_results


def do_it(userQuery):
    userVectors = get_embeddings(bedrock, userQuery)
    similaritysearchResponse = get_knn_results(oss_client, userVectors)

    keyword_1, keyword_2, keyword_3 = extract_keywords(bedrock, userQuery)

    keyword_1_results = get_keyword_results(oss_client, keyword_1)

    print(f"keyword_1_results:------------------------- {keyword_1_results}-------------------------------------------")

    keyword_2_results = get_keyword_results(oss_client, keyword_2)

    print(f"keyword_2_results:------------------------- {keyword_2_results}-------------------------------------------")

    keyword_3_results = get_keyword_results(oss_client, keyword_3)

    print(f"keyword_3_results:------------------------- {keyword_3_results}-------------------------------------------")

    #join the keyword_x_results strings into a single string with two new lines as a seperator
    full_keyword_results = keyword_1_results + "\n\n" + keyword_2_results + "\n\n" + keyword_3_results

    keyword_1_vectors = get_embeddings(bedrock, keyword_1) 

    keyword_2_vectors = get_embeddings(bedrock, keyword_2) 

    keyword_3_vectors = get_embeddings(bedrock, keyword_3)


    keyword_1_knn_results = get_knn_keyword_results(oss_client, keyword_1_vectors)

    print(f"keyword_knn_results:-------------------------- {keyword_1_knn_results} ----------------------------------")

    keyword_2_knn_results = get_knn_keyword_results(oss_client, keyword_2_vectors)

    print(f"keyword_knn_results:-------------------------- {keyword_2_knn_results} ----------------------------------")

    keyword_3_knn_results = get_knn_keyword_results(oss_client, keyword_3_vectors)

    print(f"keyword_knn_results:-------------------------- {keyword_3_knn_results} ----------------------------------")

    full_keyword_knn_results = keyword_1_knn_results + "\n\n" + keyword_2_knn_results + "\n\n" + keyword_3_knn_results

    llmOutput = invoke_llm(bedrock, userQuery, similaritysearchResponse, full_keyword_results, full_keyword_knn_results)
    return llmOutput


st.set_page_config(page_title="Compliance Q+A", page_icon=":tada", layout="wide")

#Headers
with st.container():
    st.header("Ask your Compliance Questions here")


#
with st.container():
    st.write("---")
    userQuery = st.text_input("Ask a Question")
    #userID = st.text_input("User ID")
    st.write("---")


##Back to Streamlit
result=st.button("ASK!")
if result:
    st.write(do_it(userQuery))



