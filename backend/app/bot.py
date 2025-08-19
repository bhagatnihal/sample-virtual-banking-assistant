import os
import boto3
# from dotenv import load_dotenv
from boto3.dynamodb.conditions import Key
# load_dotenv()

TABLE_NAME = os.getenv("DYNAMODB_TABLE")
REGION = os.getenv("AWS_REGION", "us-east-1")

dynamodb = boto3.resource("dynamodb", region_name=REGION)

table = dynamodb.Table(TABLE_NAME)

class RecordNotFoundError(Exception):
    pass

def get_bot_by_id(bot_id:str):
    """Find bot details from dynamodb"""
    response = table.query(
        IndexName="PublicBotIdIndex",
        KeyConditionExpression=Key("PublicBotId").eq(bot_id),
    )
    if len(response["Items"]) == 0:
        raise RecordNotFoundError(f"Bot with id {bot_id} not found.")

    item = response["Items"][0]
    
    return item