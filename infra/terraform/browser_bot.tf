# Browser Bot ECS Service Configuration

resource "aws_ecs_service" "browser_bot_service" {
  name            = "clerk-browser-bot-service"
  cluster         = aws_ecs_cluster.clerk_cluster.id
  task_definition = aws_ecs_task_definition.browser_bot.arn
  desired_count   = 0  # Start with 0, scale based on demand
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.browser_bot_sg.id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.browser_bot.arn
  }

  depends_on = [
    aws_iam_role_policy_attachment.ecs_task_execution_role_policy,
    aws_iam_role_policy_attachment.ecs_task_role_policy
  ]

  tags = {
    Name        = "clerk-browser-bot-service"
    Environment = var.environment
    Service     = "browser-bot"
  }
}

resource "aws_ecs_task_definition" "browser_bot" {
  family                   = "clerk-browser-bot"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.browser_bot_cpu
  memory                   = var.browser_bot_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn           = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name  = "browser-bot"
      image = "${var.aws_account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/clerk-browser-bot:latest"
      
      essential = true
      
      portMappings = [
        {
          containerPort = 3000
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "NODE_ENV"
          value = "production"
        },
        {
          name  = "RT_GATEWAY_URL"
          value = "wss://${var.rt_gateway_domain}"
        },
        {
          name  = "API_BASE_URL"
          value = "https://${var.api_domain}"
        },
        {
          name  = "JOIN_TIMEOUT_SEC"
          value = "60"
        },
        {
          name  = "AUDIO_SAMPLE_RATE"
          value = "16000"
        },
        {
          name  = "AUDIO_CHANNELS"
          value = "1"
        },
        {
          name  = "LOG_LEVEL"
          value = "info"
        }
      ]

      secrets = [
        {
          name      = "AWS_ACCESS_KEY_ID"
          valueFrom = "${aws_secretsmanager_secret.aws_credentials.arn}:access_key_id::"
        },
        {
          name      = "AWS_SECRET_ACCESS_KEY"
          valueFrom = "${aws_secretsmanager_secret.aws_credentials.arn}:secret_access_key::"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.browser_bot_logs.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }

      healthCheck = {
        command = [
          "CMD-SHELL",
          "node healthcheck.js || exit 1"
        ]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 40
      }

      linuxParameters = {
        initProcessEnabled = true
        sharedMemorySize   = 2147483648
        tmpfs = [
          {
            containerPath = "/tmp"
            size          = 1073741824
          },
          {
            containerPath = "/dev/shm"
            size          = 2147483648
          }
        ]
      }

      ulimits = [
        {
          name      = "nofile"
          softLimit = 65536
          hardLimit = 65536
        },
        {
          name      = "nproc"
          softLimit = 32768
          hardLimit = 32768
        }
      ]
    }
  ])

  tags = {
    Name        = "clerk-browser-bot-task"
    Environment = var.environment
    Service     = "browser-bot"
  }
}

# Security Group for Browser Bot
resource "aws_security_group" "browser_bot_sg" {
  name_prefix = "clerk-browser-bot-"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "clerk-browser-bot-sg"
    Environment = var.environment
    Service     = "browser-bot"
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "browser_bot_logs" {
  name              = "/ecs/clerk-browser-bot"
  retention_in_days = 30

  tags = {
    Name        = "clerk-browser-bot-logs"
    Environment = var.environment
    Service     = "browser-bot"
  }
}

# Service Discovery
resource "aws_service_discovery_service" "browser_bot" {
  name = "browser-bot"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.clerk.id

    dns_records {
      ttl  = 10
      type = "A"
    }

    routing_policy = "MULTIVALUE"
  }

  health_check_grace_period_seconds = 30

  tags = {
    Name        = "clerk-browser-bot-discovery"
    Environment = var.environment
    Service     = "browser-bot"
  }
}

# Auto Scaling Target
resource "aws_appautoscaling_target" "browser_bot_scaling_target" {
  max_capacity       = var.max_browser_bot_instances
  min_capacity       = 0
  resource_id        = "service/${aws_ecs_cluster.clerk_cluster.name}/${aws_ecs_service.browser_bot_service.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

# Auto Scaling Policy - Scale Out
resource "aws_appautoscaling_policy" "browser_bot_scale_out" {
  name               = "clerk-browser-bot-scale-out"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.browser_bot_scaling_target.resource_id
  scalable_dimension = aws_appautoscaling_target.browser_bot_scaling_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.browser_bot_scaling_target.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value = 70.0

    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}

# Auto Scaling Policy - Scale In
resource "aws_appautoscaling_policy" "browser_bot_scale_in" {
  name               = "clerk-browser-bot-scale-in"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.browser_bot_scaling_target.resource_id
  scalable_dimension = aws_appautoscaling_target.browser_bot_scaling_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.browser_bot_scaling_target.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value = 30.0

    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}

# SQS Queue for Bot Management
resource "aws_sqs_queue" "browser_bot_queue" {
  name                       = "clerk-browser-bot-queue"
  visibility_timeout_seconds = 300
  message_retention_seconds  = 1209600
  receive_wait_time_seconds  = 20

  tags = {
    Name        = "clerk-browser-bot-queue"
    Environment = var.environment
    Service     = "browser-bot"
  }
}

# Dead Letter Queue
resource "aws_sqs_queue" "browser_bot_dlq" {
  name = "clerk-browser-bot-dlq"

  tags = {
    Name        = "clerk-browser-bot-dlq"
    Environment = var.environment
    Service     = "browser-bot"
  }
}

# Lambda function for bot orchestration
resource "aws_lambda_function" "browser_bot_orchestrator" {
  filename         = "browser_bot_orchestrator.zip"
  function_name    = "clerk-browser-bot-orchestrator"
  role            = aws_iam_role.lambda_execution_role.arn
  handler         = "index.handler"
  source_code_hash = data.archive_file.browser_bot_orchestrator_zip.output_base64sha256
  runtime         = "python3.9"
  timeout         = 300

  environment {
    variables = {
      ECS_CLUSTER_NAME    = aws_ecs_cluster.clerk_cluster.name
      ECS_SERVICE_NAME    = aws_ecs_service.browser_bot_service.name
      ECS_TASK_DEFINITION = aws_ecs_task_definition.browser_bot.arn
      SQS_QUEUE_URL      = aws_sqs_queue.browser_bot_queue.url
    }
  }

  tags = {
    Name        = "clerk-browser-bot-orchestrator"
    Environment = var.environment
    Service     = "browser-bot"
  }
}

# Lambda function source code
data "archive_file" "browser_bot_orchestrator_zip" {
  type        = "zip"
  output_path = "browser_bot_orchestrator.zip"
  source {
    content = templatefile("${path.module}/lambda/browser_bot_orchestrator.py", {
      ecs_cluster_name    = aws_ecs_cluster.clerk_cluster.name
      ecs_service_name    = aws_ecs_service.browser_bot_service.name
      ecs_task_definition = aws_ecs_task_definition.browser_bot.arn
    })
    filename = "index.py"
  }
}

# SQS Event Source Mapping
resource "aws_lambda_event_source_mapping" "browser_bot_sqs_trigger" {
  event_source_arn = aws_sqs_queue.browser_bot_queue.arn
  function_name    = aws_lambda_function.browser_bot_orchestrator.arn
  batch_size       = 1
  maximum_batching_window_in_seconds = 5
}
