# BMO Chatbot: RAG-Enhanced AI Assistant

This repository contains a prototype for a production-ready chatbot implementation with RAG (Retrieval-Augmented Generation) capabilities. The chatbot leverages advanced AI techniques for enhanced conversational experiences and includes comprehensive deployment, monitoring, and optimization features.

![](material/image_chatbot.png)

## Author
- Juan David Vargas Mazuera (Université de Montréal, MILA - Quebec AI Institute, CHU Sainte-Justine Research Center)

## Overview

The Chatbot combines state-of-the-art language models with agentic retrieval-augmented generation to provide accurate, contextual responses. The system is designed for production deployment with enterprise-grade monitoring and performance optimization capabilities. 

### Key Features

- **RAG Architecture**: Langgraph implementation of agentic retrieval-augmented generation  
- **Containerized Deployment**: Docker and Kubernetes support for scalable deployment
- **Comprehensive Monitoring**: Prometheus and Grafana integration for real-time metrics
- **Cache memory usage**: Usage of cache memory prevents repeated expensive operations that would otherwise happen on every streamlit rerun
- **Performance Optimized**: Base code provided for finetuning using reinforcement learning from human feedback
- **LLM evaluator**: Alternative deployment of model integrating an llm evaluator to score the relevancy of the response based on the query (Version 2)
- **Context memory**: Alternative deployment of model integrating both an llm evaluator and temporal memory to remember previous interactions


## Quick Start

### Prerequisites

- Docker 
- Kubernetes cluster (Minikube for local development)
- Helm 3.x

### Basic Installation


Docker [follow installation steps](https://minikube.sigs.k8s.io/docs/start/?arch=%2Fmacos%2Farm64%2Fstable%2Fhomebrew#Service)

for Linux
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

for Linux
```bash
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64
minikube start
```


Helm, Prometheus, and Grafana [follow installation steps](https://blog.marcnuri.com/prometheus-grafana-setup-minikube)


For Linux
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

### Deployment


# Clone the repository

```bash
git clone https://github.com/juandavidvargas19/BMO_chatbot.git -b Production
```

# Set-up open-ai keys 

Go to your open-ai account and copy your open-ai key. Make sure you have balance in your account. [open-ai keys](https://platform.openai.com/api-keys)

Then, execute and following code. Use the output as the opena-ai key in the secrets.yaml file.

```bash
secrets.yaml
```

# Run both the set-up script, and the deployment script

set-up
```bash
chmod a+x setup.sh
./setup.sh
```

deployment
```bash
chmod a+x deploy.sh
./deploy.sh
```

# Open the interface

Use the URL address to open the graphical interface of the chatbot. 
![](material/image_deploy.png)


Rate each answer to proceed
![](material/image_rate.png)


Click the "Ask Another Question" there after.
![](material/image_ask_another.png)



## Results Summary

Our results demonstrate significant improvements using the MAPS architecture:

1. **Blindsight and AGL**: 
   - Blindsight: 0.97 +/- 0.02 (Z-score: 9.01)
   - AGL- High Awareness: 0.66 +/- 0.05 (Z-score: 8.20)
   - AGL- Low Awareness: 0.62 +/- 0.07 (Z-score: 15.70)


   ![](images/Perceptual_table.png)
   
2. **MinAtar SARL**: 
   - Seaquest: 3.06 +/- 0.34 (Z-score: 7.03)
   - Asterix: 4.54 +/- 1.01 (Z-score: 1.32)
   - Breakout: 8.07 +/- 0.72 (Z-score: 3.70)
   - Space Invaders: 26.80 +/- 1.59 (Z-score: 4.13)
   - Freeway: 34.20 +/- 2.83 (Z-score: 0.95)
      
   ![](images/SARL_table.png)

   ![](images/SARL_results.jpg)

4. **MinAtar SARL + continual/transfer learning**: 
   - Achieved a mean retention of 45.1% +/- 31.1% for transfer learning. Results for continual learning are still exploratory.
     ![](images/Ternary_space.png)

5. **MARL**: 
   - Commons Harvest Partnership: 34.52 +/- 0.98 (Z-score: 6.20)
   - Commons Harvest Closed: 25.21 +/- 1.06 (Z-score: 6.31)
   - Chemistry: 1.11 +/- 0.05 (Z-score: -0.91)
   - Territory Inside Out: 48.47 +/- 1.45 (Z-score: -1.92)
      
     ![](images/MARL_table.png)
     ![](images/MARL_plot.png)


## Citation

If you want to use this code, please reach out to BMO capital markets.








