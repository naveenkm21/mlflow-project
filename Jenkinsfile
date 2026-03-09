pipeline {
    agent any

    stages {

        stage('Clone Repository') {
            steps {
                git 'https://github.com/naveenkm21/mlflow-project.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Run MLflow Training') {
            steps {
                sh 'python mlflow_demo.py'
            }
        }

        stage('Run Model Test') {
            steps {
                sh 'python load_the_model.py'
            }
        }
    }
}
