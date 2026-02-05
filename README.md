## ☁️ AWS EC2 Deployment – Sentiment Analysis Application
## 📋 Project Objective

This project demonstrates the end-to-end deployment of a Python-based Sentiment Analysis application using Amazon EC2.
The objective was to move a locally developed NLP model to a production-ready cloud environment and make it accessible through a public IP address.

## 🧠 Key Skills & Concepts Covered

* Infrastructure provisioning using AWS EC2 (IaaS)

* Linux server administration on Ubuntu

* Secure remote access using SSH and key pairs

* Network configuration using AWS Security Groups

* Hosting ML/NLP applications in a cloud environment

* Running applications persistently using tmux

## 🏗️ Infrastructure Overview

* Cloud Platform: Amazon Web Services (AWS)

* Compute Service: EC2 (Elastic Compute Cloud)

* Instance Type: t2.micro (Free Tier eligible)

* Operating System: Ubuntu Server 22.04 LTS

* Framework Used: Streamlit

* Application Port: 8501

## ⚙️ Deployment Steps
1️⃣ EC2 Instance Configuration

* Launched an EC2 instance with Ubuntu OS.

* Created a .pem key pair for secure SSH authentication.

* Configured Security Group inbound rules:

     * Port 22 (SSH)

     * Port 80 (HTTP)

     * Port 8501 (Streamlit application)

2️⃣ Secure Server Access

* Connected to the EC2 instance from the local machine using SSH after setting proper key permissions.

3️⃣ Server Environment Setup

* Updated system packages.

* Installed Python, pip, virtual environment tools, Git, and tmux.

* Prepared the server to run Python-based applications.

4️⃣ Application Deployment

* Cloned the project repository onto the EC2 instance.

* Created and activated a Python virtual environment.

* Installed all required dependencies using requirements.txt.

5️⃣ Application Execution

* Used tmux to run the application in a persistent session.

* Launched the Streamlit app and bound it to port 8501.

* Verified successful access via the EC2 public IP address.

## 🛡️ Challenges & Resolution

* Issue: Application was running but not accessible externally.
* Resolution: Added a Custom TCP inbound rule for Port 8501 in the EC2 Security Group to allow external access.

## 📈 Additional Highlights

* Deployed using AWS Free Tier for cost efficiency

* Implemented isolated dependency management using virtual environments

* Gained hands-on experience with manual cloud deployment

* Improved understanding of cloud security and networking

* Demonstrated end-to-end ownership of a production-style ML project

## ✅ Outcome

The Sentiment Analysis application was successfully deployed on AWS EC2, making it accessible in a real-world cloud environment and demonstrating practical cloud deployment skills.

