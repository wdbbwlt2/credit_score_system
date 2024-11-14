# osstem_credit_score_system
osstem_credit_score_system
# Customer Credit Score System (客户信用评分系统)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0%2B-blue.svg)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive customer credit scoring system designed for the dental medical device industry, integrating sales data, transaction history, and contract performance to generate accurate credit assessments.

## Features

- **Multi-dimensional Scoring**: Evaluates customers based on basic information, transaction history, and contract performance
- **Flexible Weight System**: Customizable scoring weights for different evaluation dimensions
- **Real-time Monitoring**: Tracks score changes and triggers alerts for significant variations
- **Batch Processing**: Supports both individual and batch credit score calculations
- **Data Visualization**: Provides various statistical analysis and visual reports
- **User-friendly Interface**: Built with Tkinter for easy operation

## System Architecture

```
credit_score_system/
├── credit_score_system/
│   ├── __init__.py
│   ├── calculator.py
│   ├── core.py
│   ├── utils.py
│   └── validation.py
├── main.py
├── config.json
├── setup.py
└── test.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-credit-score-system.git
cd customer-credit-score-system
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Configure the database:
- Create a MySQL database
- Update `config.json` with your database credentials

4. Initialize the system:
```bash
python setup.py
```

## Configuration

Update `config.json` with your settings:

```json
{
    "database": {
        "host": "localhost",
        "user": "your_username",
        "password": "your_password",
        "database": "your_database",
        "port": 3306
    }
}
```

## Usage

1. Start the application:
```bash
python main.py
```

2. Use the GUI to:
- Calculate individual customer credit scores
- Process batch scoring
- Generate statistical reports
- Configure system settings

## Testing

Run the test suite:
```bash
python test.py
```

## Project Structure

- `calculator.py`: Core scoring calculation logic
- `core.py`: System initialization and management
- `utils.py`: Utility functions and database operations
- `validation.py`: Data validation and verification
- `main.py`: GUI application
- `test.py`: Test suite

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Now creating `.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Logs
logs/
*.log

# IDE
.idea/
.vscode/
*.swp
*.swo
.project
.pydevproject
.settings

# Environment
.env
.venv
env/
venv/
ENV/

# Database
*.sqlite3
*.db

# Config
config.json

# Others
.DS_Store
Thumbs.db
```

Now creating `requirements.txt`:

```
pandas>=1.3.0
numpy>=1.19.0
sqlalchemy>=1.4.0
pymysql>=1.0.0
matplotlib>=3.4.0
openpyxl>=3.0.0
```

Now creating `LICENSE`:

```
MIT License

Copyright (c) 2024 Customer Credit Score System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
