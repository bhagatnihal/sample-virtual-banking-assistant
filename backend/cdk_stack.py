# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: MIT-0

"""
AWS CDK Stack for Voice Bot

This module defines the AWS CDK stack for deploying the Voice Bot infrastructure.
It sets up all necessary AWS resources including:
- ECS Fargate service for running the backend
- Network Load Balancer for handling WebSocket connections
- Cognito User Pool for authentication
- S3 and CloudFront for frontend hosting
- IAM roles and security groups

The stack is designed to be deployed in a VPC with both public and private subnets,
supporting a secure and scalable architecture.
"""

from aws_cdk import (
    Stack,
    Duration,
    CfnOutput,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecr_assets as ecr_assets,
    aws_elasticloadbalancingv2 as elbv2,
    aws_iam as iam,
    aws_apigateway as apigateway,
    aws_apigatewayv2 as apigatewayv2,
    aws_cognito as cognito,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_certificatemanager as acm,
    RemovalPolicy,
    aws_secretsmanager as secretsmanager,
)
from aws_cdk.aws_apigatewayv2 import CfnIntegration, CfnRoute
from constructs import Construct
import cdk_nag

container_port = 8000

class CdkStack(Stack):
    """CDK Stack for Voice Bot infrastructure.
    
    This stack creates all necessary AWS resources for running the Voice Bot,
    including compute resources, networking, authentication, and frontend hosting.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        """Initialize the CDK stack.
        
        Args:
            scope: The scope in which to define this construct
            construct_id: The scoped construct ID
            **kwargs: Additional arguments to pass to Stack
        """
        super().__init__(scope, construct_id, **kwargs)

        secret = secretsmanager.Secret.from_secret_name_v2(
            self, "VoiceBotSecret", "voicebot/getBot/tablename"
        )

        # Get VPC configuration from context
        vpc_config = self.node.try_get_context('vpc-config')
        certificate_arn = self.node.try_get_context('certificate-arn')
        
        # Get reference to existing VPC and subnets
        vpc = ec2.Vpc.from_vpc_attributes(self, "VoiceBotVPC",
            vpc_id=vpc_config['vpcId'],
            availability_zones=vpc_config['availabilityZones'],
            public_subnet_ids=vpc_config['publicSubnetIds']
        )

        # Create ECS Cluster in existing VPC
        cluster = ecs.Cluster(self, "VoiceBotCluster",
            vpc=vpc,
        )

        # Build and push Docker image to ECR
        docker_image = ecr_assets.DockerImageAsset(self, "VoiceBotImage",
            directory=".",
            file="Dockerfile"
        )

        # Create Task Role with required permissions
        task_role = iam.Role(self, "VoiceBotTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="Role for Voice ECS Task"
        )
        
        # Add ECR pull permissions
        task_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy")
        )

        # Add Bedrock permissions
        task_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel"
                ],
                resources=["arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-sonic-v1:0"]
            )
        )

        # Create security group for the Fargate service
        security_group = ec2.SecurityGroup(self, "VoiceBotServiceSG",
            vpc=vpc,
            allow_all_outbound=True,
            description="Security group for FastAPI service"
        )
        security_group.add_ingress_rule(
            ec2.Peer.ipv4(vpc_config['cidr']),
            ec2.Port.tcp(container_port),
            "Allow inbound HTTP traffic from VPC only"
        )

        # Task Definition for Fargate
        task_def = ecs.FargateTaskDefinition(self, "VoiceBotTaskDef", 
            task_role=task_role,
            execution_role=task_role,
            cpu=2048,
            memory_limit_mib=4096,
        )
        container = task_def.add_container(
            "VoiceBotContainer",
            image=ecs.ContainerImage.from_docker_image_asset(docker_image),
            logging=ecs.LogDriver.aws_logs(stream_prefix="VoiceBot"),
            secrets={
                "DYNAMODB_TABLE": ecs.Secret.from_secrets_manager(
                    secret, "DYNAMODB_TABLE"
                ),
                "AWS_REGION": ecs.Secret.from_secrets_manager(secret, "AWS_REGION"),
            },
        )
        container.add_port_mappings(ecs.PortMapping(container_port=container_port, protocol=ecs.Protocol.TCP))

        # ECS Fargate Service in private subnets
        service = ecs.FargateService(
            self,
            "VoiceBotFargateService",
            cluster=cluster,
            task_definition=task_def,
            assign_public_ip=True,
            desired_count=1,
            enable_execute_command=True,
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            security_groups=[security_group],
            vpc_subnets=ec2.SubnetSelection(
                subnets=[
                    ec2.Subnet.from_subnet_id(
                        self, "TaskSubnet1", vpc_config["privateSubnetIds"][0]
                    ),
                    ec2.Subnet.from_subnet_id(
                        self, "TaskSubnet2", vpc_config["privateSubnetIds"][1]
                    ),
                ]
            ),
        )

        # Network Load Balancer in public subnets
        nlb = elbv2.NetworkLoadBalancer(
            self,
            "VoiceBotNLB",
            vpc=vpc,
            internet_facing=True,
            cross_zone_enabled=True,
            vpc_subnets=ec2.SubnetSelection(
                subnets=[
                    ec2.Subnet.from_subnet_id(
                        self, "NlbSubnet1", vpc_config["publicSubnetIds"][0]
                    ),
                    ec2.Subnet.from_subnet_id(
                        self, "NlbSubnet2", vpc_config["publicSubnetIds"][1]
                    ),
                ]
            ),
        )

        # TCP Target Group for WebSocket traffic
        target_group = elbv2.NetworkTargetGroup(
            self,
            "VoiceBotTargetGroup",
            vpc=vpc,
            port=container_port,
            protocol=elbv2.Protocol.TCP,
            target_type=elbv2.TargetType.IP,
            deregistration_delay=Duration.seconds(
                120
            ),  # Allow 2 minutes for connections to drain
            health_check=elbv2.HealthCheck(
                protocol=elbv2.Protocol.HTTP,
                path="/health",
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
                interval=Duration.seconds(30),
                timeout=Duration.seconds(10),
                healthy_http_codes="200-399",
            ),
        )

        # Attach ECS service to Target Group
        service.attach_to_network_target_group(target_group)

        # Configure NLB Listener
        listener = nlb.add_listener(
            "VoiceBotHttpListener",
            port=443 if certificate_arn else 80,
            protocol=elbv2.Protocol.TLS if certificate_arn else elbv2.Protocol.TCP,
            certificates=[elbv2.ListenerCertificate.from_arn(certificate_arn)]
            if certificate_arn
            else [],
            default_target_groups=[target_group],
        )

        # # Create Cognito User Pool
        user_pool = cognito.UserPool(self, "VoiceBotUserpool",
            removal_policy=RemovalPolicy.DESTROY,
            self_sign_up_enabled=False,
            enable_sms_role=False,
            mfa=cognito.Mfa.OFF,
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            password_policy=cognito.PasswordPolicy(
                min_length=8,
                require_digits=True,
                require_lowercase=True,
                require_symbols=True,
                require_uppercase=True
            ),
            standard_attributes=cognito.StandardAttributes(
                email=cognito.StandardAttribute(
                    required=True,
                    mutable=False
                )
            ),
            auto_verify=cognito.AutoVerifiedAttrs(
                email=True
            )
        )

        # # Add Cognito User Pool Client
        user_pool_client = user_pool.add_client("VoiceBotUserpoolClient",
            auth_flows=cognito.AuthFlow(
                admin_user_password=False,
                custom=False,
                user_password=False,
                user_srp=True
            ),
            disable_o_auth=True,
            prevent_user_existence_errors=True,
            supported_identity_providers=[]
        )

        # # Create Cognito Identity Pool
        identity_pool = cognito.CfnIdentityPool(self, "VoiceBotIdentityPool",
            allow_unauthenticated_identities=False,
            allow_classic_flow=False,
            cognito_identity_providers=[cognito.CfnIdentityPool.CognitoIdentityProviderProperty(
                client_id=user_pool_client.user_pool_client_id,
                provider_name=user_pool.user_pool_provider_name
            )]
        )

        # # Create IAM role for authenticated users
        authenticated_role = iam.Role(self, "VoiceBotAuthenticatedRole",
            assumed_by=iam.FederatedPrincipal(
                'cognito-identity.amazonaws.com',
                {
                    "StringEquals": {
                        "cognito-identity.amazonaws.com:aud": identity_pool.ref
                    },
                    "ForAnyValue:StringLike": {
                        "cognito-identity.amazonaws.com:amr": "authenticated"
                    }
                },
                'sts:AssumeRoleWithWebIdentity'
            )
        )

        # # Attach roles to identity pool
        cognito.CfnIdentityPoolRoleAttachment(self, "VoiceBotRoleAttachment",
            identity_pool_id=identity_pool.ref,
            roles={
                'authenticated': authenticated_role.role_arn
            }
        )

        # # Create S3 bucket for frontend hosting
        website_bucket = s3.Bucket(self, "VoiceBotBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            access_control=s3.BucketAccessControl.PRIVATE,
            enforce_ssl=True
        )

        # # Create CloudFront distribution
        distribution = cloudfront.Distribution(self, "VoiceBotDistribution",
            comment="Virtual Banking Assistant Frontend",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(website_bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED
            ),
            default_root_object='index.html',
            minimum_protocol_version=cloudfront.SecurityPolicyProtocol.TLS_V1_2_2021,
            error_responses=[
                cloudfront.ErrorResponse(
                    http_status=403,
                    response_http_status=200,
                    response_page_path='/index.html'
                ),
                cloudfront.ErrorResponse(
                    http_status=404,
                    response_http_status=200,
                    response_page_path='/index.html'
                )
            ]
        )

        # Output important resource information
        CfnOutput(self, "UserPoolId", value=user_pool.user_pool_id)
        CfnOutput(self, "UserPoolClientId", value=user_pool_client.user_pool_client_id)
        CfnOutput(self, "IdentityPoolId", value=identity_pool.ref)
        CfnOutput(self, "CloudFrontURL", value=f"https://{distribution.distribution_domain_name}")
        CfnOutput(self, "NLBEndpoint", value=f"https://{nlb.load_balancer_dns_name}")
        CfnOutput(self, "FrontendBucket", value=website_bucket.bucket_name)

        # cdk-nag suppressions.
        cdk_nag.NagSuppressions.add_resource_suppressions_by_path(
            self,
            f"/{self.stack_name}/VoiceBotCluster/Resource",
            [
                {
                    "id": "AwsSolutions-ECS4",
                    "reason": "Container insights wont be used for sample code.",
                }
            ],
        )

        cdk_nag.NagSuppressions.add_resource_suppressions_by_path(
            self,
            f"/{self.stack_name}/VoiceBotTaskRole/Resource",
            [
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "AmazonECSTaskExecutionRolePolicy is necessary.",
                }
            ],
        )

        cdk_nag.NagSuppressions.add_resource_suppressions_by_path(
            self,
            f"/{self.stack_name}/VoiceBotTaskRole/DefaultPolicy/Resource",
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "Wildcards are from the managed policy.",
                }
            ],
        )

        cdk_nag.NagSuppressions.add_resource_suppressions_by_path(
            self,
            f"/{self.stack_name}/VoiceBotNLB/Resource",
            [
                {
                    "id": "AwsSolutions-ELB2",
                    "reason": "Access logging wont be used for sample code.",
                }
            ],
        )

        cdk_nag.NagSuppressions.add_resource_suppressions_by_path(self, 
            f'/{self.stack_name}/VoiceBotUserpool/Resource',
            [
                {
                    'id': 'AwsSolutions-COG3',
                    'reason': 'Advaced security model wont be used for sample code.'
                }
            ]
        )

        cdk_nag.NagSuppressions.add_resource_suppressions_by_path(self, 
            f'/{self.stack_name}/VoiceBotBucket/Resource',
            [
                {
                    'id': 'AwsSolutions-S1',
                    'reason': 'S3 access logging wont be used for sample code.'
                }
            ]
        )

        cdk_nag.NagSuppressions.add_resource_suppressions_by_path(self, 
            f'/{self.stack_name}/VoiceBotDistribution/Resource',
            [
                {
                    'id': 'AwsSolutions-CFR3',
                    'reason': 'Cloudfront access logging wont be used for sample code.'
                }
            ]
        )

        cdk_nag.NagSuppressions.add_resource_suppressions_by_path(self, 
            f'/{self.stack_name}/VoiceBotDistribution/Resource',
            [
                {
                    'id': 'AwsSolutions-CFR4',
                    'reason': 'Default CloudFront viewer certificate is enough for sample code. Already uses TLS 1.2 as minimum.'
                }
            ]
        )