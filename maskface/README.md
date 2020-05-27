| model | batchsize | num_epoch | optimizer | lr_schedule | loss | acc |
| --- | --- | --- | --- | --- | --- | --- |
| resnet18 | 32 | 100 | SGD(..., lr=0.01, momentum=0.9) | StepLR(step_size=30, gamma=0.1) | 0.0024 | 98.01% |
| resnet50 | 32 | 100 | SGD(..., lr=0.01, momentum=0.9) | StepLR(step_size=30, gamma=0.1) | 0.0027 | 97.61% |
| resnet34 | 32 | 100 | SGD(..., lr=0.01, momentum=0.9) | StepLR(step_size=30, gamma=0.1) | 0.0026 | 97.96% |
| resnet34 | 256 | 100 | SGD(..., lr=0.01, momentum=0.9) | StepLR(step_size=30, gamma=0.1) | 0.0003 | 97.81% |
| resnet18 | 32 | 100 | SGD(..., lr=0.01, momentum=0.0) | StepLR(step_size=30, gamma=0.1) | 0.0023 | 97.91% |
| densenet121 | 32 | 100 | SGD(..., lr=0.01, momentum=0.9) | StepLR(step_size=30, gamma=0.1) | 0.0023 | 98.01% |
| mobilenet_v2 | 32 | 100 | SGD(..., lr=0.01, momentum=0.9) | StepLR(step_size=30, gamma=0.1) | 0.0040 | 98.37% |

