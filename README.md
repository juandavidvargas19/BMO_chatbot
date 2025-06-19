# Agentic RAG Chatbot: RAG-Enhanced AI Assistant

This repository contains a prototype for a production-ready chatbot implementation with RAG (Retrieval-Augmented Generation) capabilities. The chatbot leverages advanced AI techniques for enhanced conversational experiences and includes comprehensive deployment, monitoring, and optimization features.

<p align="center">
  <img src="material/image_chatbot.png">
</p>

## AUTHOR
- Juan David Vargas Mazuera (Université de Montréal, MILA - Quebec AI Institute, CHU Sainte-Justine Research Center)

## METHODOLOGY

The Chatbot combines state-of-the-art language models with agentic retrieval-augmented generation to provide accurate, contextual responses. The system is designed for production deployment with enterprise-grade monitoring and performance optimization capabilities. We use the design thinking methodology. Design thinking ensures we deeply understand user needs before building technical solutions, preventing the common trap of creating sophisticated AI systems that don't solve real workplace problems.

<p align="center">
  <img src="material/designthinking.jpg">
</p>

## EMPATHIZE & DEFINE

We start the design of our RAG chatbot by outlining a design brief. A design brief aligns all stakeholders on project scope, constraints, and success metrics before development begins, preventing costly scope creep and ensuring the RAG chatbot meets actual business requirements rather than technical assumptions.

<p align="center">
  <img src="material/Design_Brief.png">
</p>


## IDEATE

Based on the project goals, we ideate base prototypes that could solve the problem.

### Versions

There are 4 prototype ideas. These are:

1. **Base model**: Agentic RAG (Version 1).
<p align="center">
  <img src="material/rag_agent_graph_v1.png">
</p>

2. **Base model + quality check**: Alternative deployment of model integrating an llm evaluator to score the relevancy of the response based on the query. The evalautor is used for quality assesment and trigering a loop effect to ensure a quality response (Version 2).
<p align="center">
  <img src="material/rag_agent_graph_v2.png">
</p>

3. **Base model + context memory**: Alternative deployment of model integrating both an llm evaluator and temporal memory to remember previous interactions (Version 3).
<p align="center">
  <img src="material/rag_agent_graph_v3.png">
</p>

4. **Base model + quality check + context memory**: Alternative deployment of model integrating both an llm evaluator and temporal memory to remember previous interactions. The evalautor is used for quality assesment and trigering a loop effect to ensure a quality response (Version 4)
<p align="center">
  <img src="material/rag_agent_graph_v4.png">
</p>

### Features


- **RAG Architecture**: Langgraph implementation of agentic retrieval-augmented generation  
- **Containerized Deployment**: Docker and Kubernetes support for scalable deployment
- **Comprehensive Monitoring**: Prometheus and Grafana integration for real-time metrics
- **Cache memory usage**: Usage of cache memory prevents repeated expensive operations that would otherwise happen on every streamlit rerun
- **Performance Optimized**: Base code provided for finetuning using reinforcement learning from human feedback
- **LLM evaluator**: Alternative deployment of model integrating an llm evaluator to score the relevancy of the response based on the query (Version 2, 3 and 4)
- **Context memory**: Alternative deployment of model integrating both an llm evaluator and temporal memory to remember previous interactions (Version 3)
- **Quality check**: The llm evalautor is used for quality assesment and trigering a loop effect to ensure a quality response (Version 2 and 4)

### Expected tradeOffs

<p align="center">
  <img src="material/tradeoffs_table.png">
</p>

## PROTOTYPE

![](material/image_structure_code.png)


### Prerequisites

- Docker 
- Kubernetes cluster (Minikube for local development)
- Helm 
- Prometheus and Grafana


Docker [follow installation steps](https://minikube.sigs.k8s.io/docs/start/?arch=%2Fmacos%2Farm64%2Fstable%2Fhomebrew#Service)

```bash
#Set up Docker's apt repository.
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
# Add the repository to Apt sources:
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \ sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
# Install the Docker packages.
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
# Verify that the installation is successful by running the hello-world image:
sudo usermod -aG docker $USER
newgrp docker
sudo docker run hello-world
```


Minikube [follow installation steps](https://minikube.sigs.k8s.io/docs/start/?arch=%2Fmacos%2Farm64%2Fstable%2Fhomebrew#Service)

```bash
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64
minikube start
```


Helm, Prometheus, and Grafana [follow installation steps](https://blog.marcnuri.com/prometheus-grafana-setup-minikube)


```bash
#install helm, this is pre requiesite for both prometheus and grafana
sudo snap install helm --classic
```

```bash
#installation steps for prometheus
# read more in https://blog.marcnuri.com/prometheus-grafana-setup-minikube
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/prometheus
kubectl expose service prometheus-server --type=NodePort --target-port=9090 --name=prometheus-server-np
kubectl get pods -l app.kubernetes.io/instance=prometheus # (optional) check whether everything has been deployed
minikube service prometheus-server-np # to open prometheus web interface

```

```bash
#installation steps for grafana
helm repo add grafana https://grafana.github.io/helm-charts
helm install grafana grafana/grafana
kubectl expose service grafana --type=NodePort --target-port=3000 --name=grafana-np
kubectl get secret --namespace default grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo #get grafana admin password
```

Follow the instructions in the command, save the credentials, and enter Grafana's interface using the key from the previous step:

```bash
minikube service grafana-np #load grafana web interface using credentials
# user = admin
# password = output from previous command
```


### Clone the repository

```bash
git clone https://github.com/juandavidvargas19/Production_RAG_chatbot.git -b Beta
```

### Set-up open-ai keys 

Go to your open-ai account and copy your open-ai key. Make sure you have balance in your account. [open-ai keys](https://platform.openai.com/api-keys)

Then, execute and following code. Use the output as the opena-ai key in the secrets.yaml file.

```bash
echo -n 'sk-your-actual-api-key' | base64
```

### Run both the set-up script, and the deployment script

set-up
```bash
chmod a+x setup.sh
./setup.sh
```

For the deployment, there are 4 versions available that can be deployed depending on the specific needs of the client. To deploy each, you need to run the deployment script as in the example:

```bash
chmod a+x deploy.sh
./deploy.sh 1 #version 1
# ./deploy.sh 2  #version 2
# ./deploy.sh 3 #version 3
# ./deploy.sh 4 #version 4
```

### Open the interface

Use the URL address to open the graphical interface of the chatbot. 

<p align="center">
  <img src="material/image_deploy.png">
</p>

Rate each answer to proceed
<p align="center">
  <img src="material/image_rate.png">
</p>

Click the "Ask Another Question" there after.
<p align="center">
  <img src="material/image_ask_another.png">
</p>


### Monitoring

To succesfully set up your monitoring pipeline you need to follow 4 steps:

1. Enter the Grafana Graphical interface using the URL address displayed
<p align="center">
  <img src="material/image_deploy.png">
</p>

2. Add prometheus as a data source
<p align="center">
  <img src="material/image_data_source.png">
</p>

3. Import the json file [json file](https://github.com/juandavidvargas19/Production_RAG_chatbot/tree/Production/Monitoring_Template)
<p align="center">
  <img src="material/image_import_dashboard.png">
</p>

4. Enjoy
<p align="center">
  <img src="material/image_dashboard.png">
</p>

### Optimization

To leverage the potential of agentic RAG, we implement a pipeline to finetune the main llm using reinforcement learning from human feedback. 

Our target score is the average of 2 metrics: (normalized user score per dollar, normalized context adherence per dollar). This results in a score in the range [0.00 , 1.00]. In the pipeline, we find a reward model using these metrics, to then finetune our llm. The result is model id which we can change in "Langgraph_Agent.py" file.  The result will look like this:

```bash
{
  "model_id": "ft:gpt-4o-2024-08-06:personal::Bew7X3c0",
  "job_id": "ftjob-d6IPhvlakLgoW9Z0i6czxsTv",
  "created_at": "2025-06-04T23:04:21.227879",
  "training_file": "fine_tuning_data_20250604_225715.jsonl",
  "base_model": "gpt-4o-2024-08-06"
}
```

To run this pipeline, you need to collect a big sample of interactions with the chatbot. At every interaction, the necesary inputs will be saved in a "training_data.jsonl" file. We will copy this file to the RLHF directory, and then run the script to finetune our llm.

```bash
cp training_data.jsonl RLHF
python RLHF.py
```

If you executed your session with Minikube, you can extract the jsonl file with

```bash
# Append new lines from pod to existing local file
kubectl exec $(kubectl get pods -l app=pdf-rag-app -o jsonpath='{.items[0].metadata.name}') -- cat /app/training_data.jsonl >> training_data_kubernetes.jsonl

#OPTION 2, download the file
#kubectl cp $(kubectl get pods -l app=pdf-rag-app -o jsonpath='{.items[0].metadata.name}'):/app/training_data.jsonl training_data_kubernetes.jsonl
```


## TEST

### Generic prompts

We used 10 generic prompts (generated by chatgpt, questions 1-5, and claude, questions 6-10) as to test the performance metrics of the 3 versions implemented. The prompts are as follow:


#### 1. **Conceptual Understanding**
**Q:** *What is the bias-variance tradeoff in machine learning, and how does it impact model performance?*

#### 2. **Practical Implementation**
**Q:** *How would you implement early stopping in a PyTorch training loop, and why is it beneficial?*

#### 3. **Recent Research**
**Q:** *Summarize the main contributions of the "LoRA" technique for fine-tuning large language models.*

####  4. **Comparison & Evaluation**
**Q:** *Compare XGBoost and Random Forest in terms of performance, interpretability, and training time on tabular datasets.*

#### 5. **Edge Cases & Recommendations**
**Q:** *If a dataset contains severe class imbalance (e.g., 1% positive class), what are three strategies to improve classifier performance, and when would each be preferred?*

####  6. **Conceptual Understanding + Application**
**Q:** *What is the difference between bagging and boosting ensemble methods? Can you provide a specific example of when you would choose Random Forest over AdaBoost for a real-world problem?*

####  7. **Technical Implementation + Best Practices**
**Q:** *How do you handle class imbalance in a binary classification problem with a 95:5 ratio? Explain at least three different approaches and their trade-offs, including when to use each method.*

####  8. **Mathematical Foundation + Intuition**
**Q:** *Explain the mathematical intuition behind why gradient descent works for neural network optimization. What happens to the loss landscape when you add L2 regularization?*

####  9. **Practical Troubleshooting + Debugging**
**Q:** *My neural network is overfitting on the training data despite using dropout and early stopping. The validation loss starts increasing after epoch 10, but training loss keeps decreasing. What are the possible causes and solutions?*

####  10. **Advanced Topics + Current Trends**
**Q:** *Compare transformer attention mechanisms with CNN feature extraction for computer vision tasks. In what scenarios would you choose Vision Transformers (ViTs) over traditional CNNs, and what are the computational trade-offs?*

### Evaluation table

<p align="center">
  <img src="material/Evaluation_table.png">
</p>

## REPORT

To refer to our full production plans, as well as answers to relevant question about the implementation of this chatbot, please open the plannification file referenced. [open](https://github.com/juandavidvargas19/Production_RAG_chatbot/tree/Production/material/RAG_Chatbot_Report.pdf)


## DEMO VIDEO

Check a demo [here](https://youtu.be/ogxsstCcdrQ)








