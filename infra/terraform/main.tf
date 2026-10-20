# Skeleton for the production footprint (spec 5.2: ~20-23 self-hosted GPUs). Fill in provider details.
# Pools: T0 (encoders/embeddings, small GPUs), T1a FIM (isolated), T1b code (~30B FP8), T1c/T2 (8B), CPU for the API.
terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.70" }
  }
}

variable "region" { default = "ap-south-1" }
variable "cluster_name" { default = "helix-prod" }

variable "gpu_pools" {
  description = "Node groups per model tier (instance type, min, max). Sized from the capacity model."
  type        = map(object({ instance_type = string, min = number, max = number }))
  default = {
    t0_encoders = { instance_type = "g5.xlarge", min = 2, max = 4 }      # L1, L2, L3 on TEI / Triton
    t1a_fim     = { instance_type = "p4d.24xlarge", min = 1, max = 2 }   # ~7 x A100 for ~375 req/s completion
    t1b_code    = { instance_type = "p5.48xlarge", min = 1, max = 1 }    # ~30B code model FP8, multi-LoRA
    t1c_t2      = { instance_type = "g6e.12xlarge", min = 2, max = 4 }   # 8B models, L6/L7 adapters
  }
}

provider "aws" { region = var.region }

# module "eks" { source = "terraform-aws-modules/eks/aws" ... }
# resource "aws_eks_node_group" "gpu" { for_each = var.gpu_pools ... taints for tier isolation ... }
# KEDA ScaledObjects scale vLLM deployments on queue depth; the completion pool never shares GPUs with long-context work.

output "gpu_pools" { value = var.gpu_pools }
