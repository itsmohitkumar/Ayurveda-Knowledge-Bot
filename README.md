# Indian Tax Advisor Application

## Overview

The Indian Tax Advisor is a web application designed to provide personalized tax advice to users in India. The application features a FastAPI backend that processes queries related to tax deductions, savings strategies, and general inquiries about the Indian tax system. The React frontend offers a user-friendly interface for users to interact with the application.

This project can be easily deployed using Docker, allowing for streamlined development and production environments.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Docker](#docker)
- [Contributing](#contributing)
- [License](#license)

## Features

- **FastAPI Backend**: A robust and efficient backend handling tax-related queries.
- **React Frontend**: A modern and interactive user interface built with React.
- **AWS Integration**: Users can configure AWS API keys for backend functionality.
- **Docker Support**: Simplified deployment and management using Docker containers.
- **User-Friendly Design**: Intuitive layout and responsive design for optimal user experience.

## Technologies Used

- **Python 3.11.x**: Backend development with FastAPI.
- **Node.js 18.x**: Frontend development with React.
- **Tailwind CSS**: Styling framework for a clean and modern UI.
- **Docker**: For containerization of the application, ensuring consistency across environments.

## Project Structure

```plaintext
project-root/
│
├── app.py              # Main entry point for the FastAPI application
├── requirements.txt     # Python dependencies for the backend
├── src/chatbot/        # Source code for the chatbot functionality
│   ├── config.py       # Configuration settings for the chatbot
│   └── ...
│
├── frontend-react/      # Source code for the React frontend
│   ├── src/            # Main source files for the React app
│   ├── package.json     # Node dependencies for the React app
│   ├── public/          # Static files for the frontend
│   └── ...
│
├── Dockerfile           # Dockerfile for containerization of the application
└── README.md            # Project documentation
```

## Installation

### Prerequisites

- **Docker**: Ensure Docker is installed on your machine. Refer to the [Docker installation guide](https://docs.docker.com/get-docker/) for details.
- **Python**: Version 3.11.x.
- **Node.js**: Version 18.x.

### Clone the Repository

```bash
git clone https://github.com/itsmohitkumar/Ayurveda-Knowledge-Bot.git
cd Ayurveda-Knowledge-Bot
```

### Backend Setup

1. Navigate to the project root.
2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

### Frontend Setup

1. Navigate to the `frontend-react` directory:

   ```bash
   cd frontend-react
   ```

2. Install the required Node packages:

   ```bash
   npm install
   ```

## Usage

### Running the Backend

1. In the project root, run the FastAPI backend:

   ```bash
   python app.py
   ```

### Running the Frontend

1. In the `frontend-react` directory, start the React application:

   ```bash
   npm start
   ```

2. Access the application in your web browser at `http://localhost:3000`.

## Docker

To run both the backend and frontend using Docker, follow these steps:

1. Build the Docker image:

   ```bash
   docker build -t indian-tax-advisor .
   ```

2. Run the Docker container:

   ```bash
   docker run -p 8000:8000 -p 3000:3000 indian-tax-advisor
   ```

3. Access the application at `http://localhost:3000`.

## Contributing

Contributions are welcome! If you have suggestions for improvements or features, please open an issue or submit a pull request. Please follow the code of conduct outlined in this repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.