// IRIS Mindmap Data Structure
const mindmapData = {
    id: "root",
    name: "IRIS Medical\nSegmentation",
    type: "root",
    level: 0,
    description: "Interactive and Refined Image Segmentation for few-shot medical image analysis",
    details: "IRIS is a state-of-the-art framework for medical image segmentation that leverages in-context learning to achieve high accuracy with minimal annotated data.",
    references: "[1, 2, 3, 4, 5]",
    children: [
        {
            id: "purpose",
            name: "Purpose",
            type: "purpose",
            level: 1,
            description: "In-Context Learning for Few-Shot Segmentation",
            details: "IRIS enables flexible adaptation to novel medical tasks through in-context learning, reducing annotation burden and enabling rapid prototyping.",
            references: "[1, 2, 3, 4, 5]",
            metrics: {
                "Goal": "Flexible adaptation to novel medical tasks",
                "Value": "Reduces annotation burden significantly",
                "Paradigm": "Conditioned on Support Set S"
            },
            children: [
                {
                    id: "goal",
                    name: "Goal",
                    level: 2,
                    description: "Flexible adaptation to novel medical tasks",
                    details: "Enable rapid deployment to new medical imaging tasks without extensive retraining or large labeled datasets.",
                    references: "[1, 2]"
                },
                {
                    id: "value",
                    name: "Value",
                    level: 2,
                    description: "Reduces annotation burden & enables rapid prototyping",
                    details: "Clinical annotation is expensive and time-consuming. IRIS achieves strong performance with just 1-5 annotated examples per class.",
                    references: "[3, 4]",
                    metrics: {
                        "Annotation Reduction": "90%+ vs fully supervised",
                        "Time to Deploy": "Minutes vs weeks"
                    }
                },
                {
                    id: "paradigm",
                    name: "Paradigm",
                    level: 2,
                    description: "Predicts masks conditioned on Support Set S",
                    details: "Uses episodic training where each iteration samples a support set (K examples) and query image, simulating few-shot deployment scenarios.",
                    references: "[5]"
                }
            ]
        },
        {
            id: "architecture",
            name: "Architecture",
            type: "architecture",
            level: 1,
            description: "5 Core Components working in harmony",
            details: "IRIS architecture consists of five specialized modules: Encoder, Task Encoding, Decoder, Memory Bank, and In-Context Tuner.",
            references: "[6-38]",
            children: [
                {
                    id: "encoder",
                    name: "3D UNet\nEncoder",
                    level: 2,
                    description: "Medical 3D UNet Encoder",
                    details: "Extracts hierarchical multi-scale features using Residual Blocks and Instance Normalization. Processes both support and query images through shared weights.",
                    references: "[6-12]",
                    metrics: {
                        "Function": "Multi-scale feature extraction",
                        "Architecture": "Residual Blocks + InstanceNorm",
                        "Output": "Features Fs, Fq for Task Encoder/Decoder"
                    },
                    children: [
                        {
                            id: "encoder-function",
                            name: "Function",
                            level: 3,
                            description: "Hierarchical multi-scale feature extraction",
                            details: "Encoder processes images at multiple resolutions (1/2, 1/4, 1/8, 1/16) to capture both fine details and global context.",
                            references: "[6, 7]"
                        },
                        {
                            id: "encoder-details",
                            name: "Details",
                            level: 3,
                            description: "Residual Blocks with Instance Normalization",
                            details: "Uses 3D residual connections for gradient flow and Instance Normalization for stable training across different imaging modalities.",
                            references: "[8-10]"
                        },
                        {
                            id: "encoder-output",
                            name: "Output",
                            level: 3,
                            description: "Features Fs, Fq to TEM & Decoder",
                            details: "Produces support features Fs and query features Fq that feed into Task Encoding Module and Bidirectional Decoder.",
                            references: "[11, 12]"
                        }
                    ]
                },
                {
                    id: "tem",
                    name: "Task Encoding\nModule",
                    level: 2,
                    description: "Task Encoding Module (TEM)",
                    details: "Encodes support set information into task embeddings. Combines foreground-focused and contextual encoding strategies.",
                    references: "[13-21]",
                    metrics: {
                        "Input": "1-3 support images + masks",
                        "Technique": "Masked pooling + Pixel Shuffle",
                        "Output": "Task Embeddings T (m+1 tokens/class)"
                    },
                    children: [
                        {
                            id: "tem-input",
                            name: "Input",
                            level: 3,
                            description: "1-3 support images with binary masks",
                            details: "Accepts K-shot support set where K typically ranges from 1 to 3 examples per class.",
                            references: "[13, 14]"
                        },
                        {
                            id: "tem-foreground",
                            name: "Foreground\nEncoding",
                            level: 3,
                            description: "Masked average pooling for fine details",
                            details: "Uses upsampled features and masked average pooling to focus on foreground regions specified by support masks.",
                            references: "[15-17]"
                        },
                        {
                            id: "tem-contextual",
                            name: "Contextual\nEncoding",
                            level: 3,
                            description: "Pixel Shuffle + Learnable Query Tokens (m=8)",
                            details: "Employs Pixel Shuffle for resolution enhancement and m=8 learnable query tokens for capturing diverse contextual patterns.",
                            references: "[15, 18-20]"
                        },
                        {
                            id: "tem-output",
                            name: "Output",
                            level: 3,
                            description: "Task Embeddings T (m+1 tokens per class)",
                            details: "Generates m+1 tokens per class: 1 foreground token + m contextual tokens, creating rich task representations.",
                            references: "[19, 21]"
                        }
                    ]
                },
                {
                    id: "decoder",
                    name: "Bidirectional\nDecoder",
                    level: 2,
                    description: "Bidirectional Mask Decoder",
                    details: "Performs bidirectional cross-attention between query features and task embeddings. Uses FiLM modulation for task conditioning.",
                    references: "[22-26]",
                    metrics: {
                        "Mechanism": "Bidirectional Cross-Attention (BCA)",
                        "Conditioning": "FiLM Modulation",
                        "Output": "Pixel-wise segmentation predictions"
                    },
                    children: [
                        {
                            id: "decoder-mechanism",
                            name: "Mechanism",
                            level: 3,
                            description: "Bidirectional Cross-Attention for feature/token exchange",
                            details: "BCA enables information flow in both directions: query features attend to task tokens AND task tokens attend to query features.",
                            references: "[22-24]"
                        },
                        {
                            id: "decoder-conditioning",
                            name: "Conditioning",
                            level: 3,
                            description: "FiLM Modulation at each decoder stage",
                            details: "Feature-wise Linear Modulation (FiLM) applies affine transformations conditioned on task embeddings at each decoding stage.",
                            references: "[22, 23, 25]"
                        },
                        {
                            id: "decoder-output",
                            name: "Output",
                            level: 3,
                            description: "Pixel-wise segmentation in single forward pass",
                            details: "Produces final segmentation mask with one forward pass, no iterative refinement needed.",
                            references: "[22, 26]"
                        }
                    ]
                },
                {
                    id: "memory",
                    name: "Memory\nBank",
                    level: 2,
                    description: "Class Memory Bank",
                    details: "Stores learned prototypes across training episodes using Exponential Moving Average (EMA). Enables zero-shot retrieval.",
                    references: "[27-34]",
                    metrics: {
                        "Update Method": "EMA (α=0.999)",
                        "Capability": "Zero-shot retrieval",
                        "Speed Boost": "35ms inference time"
                    },
                    children: [
                        {
                            id: "memory-function",
                            name: "Function",
                            level: 3,
                            description: "Stores learned class prototypes",
                            details: "Maintains a bank of class prototypes accumulated across all training episodes for knowledge preservation.",
                            references: "[27-29]"
                        },
                        {
                            id: "memory-update",
                            name: "Update",
                            level: 3,
                            description: "EMA with momentum α=0.999",
                            details: "Exponential Moving Average update: M_new = α × M_old + (1-α) × M_current ensures stable, slowly-evolving prototypes.",
                            references: "[27, 30-32]"
                        },
                        {
                            id: "memory-use",
                            name: "Use",
                            level: 3,
                            description: "Zero-shot retrieval for known objects",
                            details: "For previously seen classes, can segment new images without support set by retrieving stored prototypes. Fastest inference mode (35ms).",
                            references: "[27, 33, 34]"
                        }
                    ]
                },
                {
                    id: "tuner",
                    name: "In-Context\nTuner",
                    level: 2,
                    description: "In-Context Tuner",
                    details: "Enables rapid task adaptation by optimizing only task embeddings while freezing model weights. Requires just 5-20 gradient steps.",
                    references: "[24, 35-38]",
                    metrics: {
                        "Adaptation Speed": "5-20 gradient steps",
                        "Parameters Updated": "Task embeddings only",
                        "Performance Gain": "+7.1% Dice over one-shot"
                    },
                    children: [
                        {
                            id: "tuner-function",
                            name: "Function",
                            level: 3,
                            description: "Rapid adaptation to new tasks",
                            details: "Allows quick deployment to new medical imaging tasks or domains with minimal computation.",
                            references: "[35, 36]"
                        },
                        {
                            id: "tuner-mechanism",
                            name: "Mechanism",
                            level: 3,
                            description: "Optimizes task embeddings only (model frozen)",
                            details: "Freezes encoder/decoder weights and optimizes only the task embedding vectors via gradient descent on support set.",
                            references: "[24, 35, 37, 38]"
                        }
                    ]
                }
            ]
        },
        {
            id: "inference",
            name: "Inference\nStrategies",
            type: "inference",
            level: 1,
            description: "4 Strategies for accuracy/speed trade-offs",
            details: "IRIS offers four inference modes optimized for different deployment scenarios, from ultra-fast one-shot to highest-accuracy in-context tuning.",
            references: "[13, 33, 39-45]",
            children: [
                {
                    id: "one-shot",
                    name: "One-Shot",
                    level: 2,
                    description: "One-Shot Segmentation (K=1)",
                    details: "Fastest baseline mode using single support example. Ideal for rapid prototyping and initial testing.",
                    references: "[13, 39-41]",
                    metrics: {
                        "Support Images": "K=1",
                        "Speed": "52ms (fastest routine)",
                        "Use Case": "Quick predictions"
                    }
                },
                {
                    id: "ensemble",
                    name: "Ensemble",
                    level: 2,
                    description: "Context Ensemble (K=3)",
                    details: "Recommended balance between accuracy and speed. Averages predictions from K=3 support examples.",
                    references: "[13, 39, 40, 42]",
                    metrics: {
                        "Support Images": "K=3",
                        "Gain": "+3.6% Dice over K=1",
                        "Recommendation": "Default for most tasks"
                    }
                },
                {
                    id: "retrieval",
                    name: "Memory\nRetrieval",
                    level: 2,
                    description: "Object Retrieval from Memory Bank",
                    details: "Zero-shot capability for known classes using stored prototypes. Fastest overall inference.",
                    references: "[33, 43, 44]",
                    metrics: {
                        "Support Required": "None (zero-shot)",
                        "Speed": "35ms (fastest overall)",
                        "Limitation": "Known classes only"
                    }
                },
                {
                    id: "tuning",
                    name: "In-Context\nTuning",
                    level: 2,
                    description: "In-Context Tuning (highest accuracy)",
                    details: "Highest accuracy mode via 20 gradient steps. Recommended for critical clinical applications.",
                    references: "[39, 42, 45]",
                    metrics: {
                        "Tuning Steps": "20 recommended",
                        "Gain": "+7.1% Dice over K=1",
                        "Use Case": "Critical tasks, new domains"
                    }
                }
            ]
        },
        {
            id: "validation",
            name: "Validation",
            type: "validation",
            level: 1,
            description: "Performance across 9 diverse datasets",
            details: "IRIS validated on 9 medical imaging datasets spanning 6 modalities. Achieves competitive performance with SOTA methods while being more efficient.",
            references: "[13, 45-53]",
            children: [
                {
                    id: "datasets",
                    name: "Datasets",
                    level: 2,
                    description: "9 diverse medical imaging datasets",
                    details: "Chest X-Ray (1,400), ISIC (2,357), Brain Tumor (250), DRIVE (40), Kvasir (1,000), AMOS (240), SegTHOR (40), COVID-19 CT (200), MSD Pancreas (281)",
                    references: "[45-49]",
                    metrics: {
                        "Count": "9 datasets",
                        "Total Images": "5,805 images/volumes",
                        "Clinical Domains": "Radiology, Dermoscopy, Ophthalmology"
                    }
                },
                {
                    id: "modalities",
                    name: "Modalities",
                    level: 2,
                    description: "6 Imaging Modalities Supported",
                    details: "CT, MRI, X-Ray, Dermoscopy, Endoscopy, Fundoscopy - demonstrates cross-modality generalization.",
                    references: "[13, 50, 51]",
                    metrics: {
                        "CT": "AMOS, COVID-19, SegTHOR, Pancreas",
                        "MRI": "Brain Tumor",
                        "X-Ray": "Chest X-Ray",
                        "Dermoscopy": "ISIC",
                        "Endoscopy": "Kvasir",
                        "Fundoscopy": "DRIVE"
                    }
                },
                {
                    id: "results",
                    name: "Key Results",
                    level: 2,
                    description: "Competitive with SOTA, more efficient",
                    details: "85.1% mean Dice across 9 datasets. Outperforms nnU-Net baseline by +2.8% Dice while being 46% more energy-efficient.",
                    references: "[41, 45, 52-54]",
                    metrics: {
                        "Mean Dice": "85.1% (Ensemble K=3)",
                        "vs nnU-Net": "+2.8% Dice improvement",
                        "Efficiency": "46% more energy-efficient",
                        "Model Size": "~11.6M parameters"
                    }
                }
            ]
        },
        {
            id: "implementation",
            name: "Implementation",
            type: "implementation",
            level: 1,
            description: "PyTorch implementation details",
            details: "Built on PyTorch 2.0+ with CUDA 11.8+ for GPU acceleration. Includes episodic training loop and comprehensive documentation.",
            references: "[26, 34, 46, 54-60]",
            children: [
                {
                    id: "codebase",
                    name: "Codebase",
                    level: 2,
                    description: "PyTorch 2.0+, CUDA 11.8+",
                    details: "Modern PyTorch implementation with full GPU support, mixed precision training, and optimized data loading.",
                    references: "[46, 55]"
                },
                {
                    id: "training",
                    name: "Training",
                    level: 2,
                    description: "Episodic training loop",
                    details: "Each iteration samples K support examples + 1 query from randomly selected dataset, simulating few-shot deployment.",
                    references: "[34, 56, 57]"
                },
                {
                    id: "loss",
                    name: "Loss Function",
                    level: 2,
                    description: "Combined Loss: L_Dice + λ L_BCE (λ=1.0)",
                    details: "Optimized using combined Dice loss and Binary Cross-Entropy with equal weighting (λ=1.0) for stable training.",
                    references: "[26, 54, 58]"
                },
                {
                    id: "docs",
                    name: "Documentation",
                    level: 2,
                    description: "Architecture Guide 100% Complete",
                    details: "Comprehensive documentation includes Architecture Guide, Workflow Guide, Dataset Analysis, and Training Guide.",
                    references: "[59, 60]",
                    metrics: {
                        "Architecture Guide": "2,711 lines (complete)",
                        "Workflow Guide": "4,200 lines (complete)",
                        "Dataset Analysis": "Complete",
                        "Training Guide": "Available"
                    }
                }
            ]
        }
    ]
};
