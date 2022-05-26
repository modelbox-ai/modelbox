# ModelBox

ModelBox is an AI application development framework featuring device-edge-cloud synergy. It provides a parallel execution framework based on pipelines, helping developers quickly develop high-performance AI applications that support software-hardware synergized optimization. [See details](http://modelbox-ai.com/modelbox-book/)

## ModelBox Highlights

1. **Easy to develop**  
   Simplified orchestration and development of inference applications via a graphical interface, modularized functions, rich component libraries, and multi-language support (C++, Python).

1. **Easy to integrate**  
   Easy to integrate different components on the cloud.

1. **High performance and reliability**  
   Parallel pipeline execution, intelligent scheduling of compute capacities, fine-grained resource management and scheduling, and higher efficiency.

1. **Heterogeneous software and hardware**  
   Support for heterogeneous compute resources, including CPU, GPU and NPU, higher resource utilization.

1. **All-scenario**  
   Able to process various types of data, such as video, voice, text, and NLP; service-oriented; easy to customize and integrate; and seamless data exchange across the cloud, edge, and devices.

1. **Easy to maintain**  
   Real-time monitoring of service status and application and component performance, facilitated optimization.

## Tasks Facilitated by ModelBox

With typical AI application development, after model training, multiple models need to be joined together through coding to form a single application and the released as an online service or application. This may involve complex application programming, as described in the table below:
  
|Task|Description|
|--|--|
|Developing dependent functions for AI applications|AI application compiler project, application initialization, configuration management interface, log management interface, application fault monitoring, and more.|
|Developing common pre- and postprocessing functions for AI applications|Audion and video codecs, image conversion, preprocessing, postprocessing (YOLO), and more.|
|Enabling interconnection with cloud services|For example, HTTP service and interconnection with cloud storage, big data service, and video collection service.|
|Developing AI applications for high-performance inference|Develop applications by leveraging techniques such multi-threading, memory pooling, GPU pooling, multi-GPU accelerator, batch model processing, and hardware module calling via APIs.|
|Developing and verifying Docker images|Develop Docker images, integrate the needed software, such as FFmpeg, OpenCV, CUDA, MindSpore, and TensorFlow, and perform integration and verification tests.|
|Reusing code between different services to si|Code may need to be reused between different components, including those for preprocessing, postprocessing, and the management of applications, bottom-layer memory, and threads.|
|Verifying models|Developers may need to write a piece of Python code th verify the models they develop. To prepare the models for demanding production scenarios, the model code may still need to be rewritten or modified significantly.|

ModelBox simplifies AI application development for developers by freeing them from complex data processing, decision-making on concurrency and mutual exclusion, multi-device collaboration, code reuse between different components, data communication, and more. This way, the developers can focus on the applications themselves, rather than the underlying software details. Additionally, ModelBox also ensures software performance, reliability, and security.

## Getting Started

ModelBox can run in either of the following modes: service-oriented and SDK.

|Development Mode|Description|
|--|--|
|Service-oriented|ModelBox is offered as an independent service that helps developers develop AI application. It provides service-based components for backend services, O&M tools, and Docker images.|
|SDK|ModelBox provides development libraries for developers to extend and scale their applications for more performance-demanding inference needs, C++ and Python are supported.|

To develop an AI application for inference, follow the procedures described in [First Application](http://modelbox-ai.com/modelbox-book/develop/first-app/first-app.html).
