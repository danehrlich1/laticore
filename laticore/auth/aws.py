import bson
import boto3

def assume_role(external_id:str, role_arn:str, role_session_name:str = None,
    duration_seconds:int = 1200, boto_session:boto3.Session = None):
    """
    Uses boto3 sts client to assume role in customer's cloud.

    Args:
        external_id (str, required): external id attached to the role we're going to assume
        role_arn (str, required): the ARN of the role we're going to assume
        role_session_name (str, optional): Name of the session we're going to create
        duration_seconds (int, optional): Duration in seconds the session will be valid. Default is 1200
        boto_session (boto3.Session, optional): The first party boto session that authenticates application to konture's cloud

    Returns:
        boto_session (boto3.Session): An assumed boto3 session in the customer's cloud
    """
    if not role_session_name:
        role_session_name = str(bson.ObjectId())

    if not boto_session:
        boto_session = boto3.Session()

    sts = boto_session.client("sts")

    # make assume role call
    response = sts.assume_role(
        ExternalId = external_id,
        RoleArn = role_arn,
        RoleSessionName = role_session_name,
        DurationSeconds = duration_seconds,
    )

    # set boto session instance with credentials returned from api call
    return boto3.Session(
        aws_access_key_id =     response['Credentials']['AccessKeyId'],
        aws_secret_access_key = response['Credentials']['SecretAccessKey'],
        aws_session_token =     response['Credentials']['SessionToken'],
        )
