#!/usr/bin/env python3
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: MIT-0

"""
AWS CDK Application Entry Point

This module serves as the entry point for the AWS CDK application. It initializes and
configures the CDK stack for deployment. The stack can be environment-agnostic or
configured for specific AWS accounts and regions.

The application uses environment variables CDK_DEFAULT_ACCOUNT and CDK_DEFAULT_REGION
to determine the deployment target, making it flexible for different deployment
environments.

Usage:
    cdk deploy       # Deploy the stack to AWS
    cdk synth       # Synthesize CloudFormation template
    cdk destroy     # Remove the stack from AWS

Environment Variables:
    CDK_DEFAULT_ACCOUNT: AWS account ID for deployment
    CDK_DEFAULT_REGION: AWS region for deployment
"""

import os

import aws_cdk as cdk
from cdk_nag import AwsSolutionsChecks

from cdk_stack import CdkStack

# Initialize the CDK application
app = cdk.App()

# Add the cdk-nag AwsSolutions Pack with extra verbose logging enabled
cdk.Aspects.of(app).add(AwsSolutionsChecks(verbose=True))

# Create the Virtual Banking Assistant stack
# The stack can be environment-agnostic (deploy anywhere) or environment-specific
CdkStack(
    app,
    "VoiceBotCdkStack",
    # If you don't specify 'env', this stack will be environment-agnostic.
    # Account/Region-dependent features and context lookups will not work,
    # but a single synthesized template can be deployed anywhere.
    # For environment-specific deployment, use the current CLI configuration
    env=cdk.Environment(
        account=os.getenv("CDK_DEFAULT_ACCOUNT"), region=os.getenv("CDK_DEFAULT_REGION")
    ),
    description="Voice Bot using Pipecat and NovaSonic",
    # For explicit account/region deployment, uncomment and modify:
    # env=cdk.Environment(account='123456789012', region='us-east-1'),
    # For more information on environments, see:
    # https://docs.aws.amazon.com/cdk/latest/guide/environments.html
)

# Optional: Add AWS Solutions security checks
# Uncomment to enable detailed security analysis during synthesis
# cdk.Aspects.of(app).add(AwsSolutionsChecks(verbose=True))

# Synthesize the CloudFormation template
app.synth()
