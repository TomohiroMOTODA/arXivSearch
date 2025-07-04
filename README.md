# ArXiv Robotics Research Trend Analyzer

A Python tool for analyzing research trends in robotics foundation models and Vision-Language-Action (VLA) papers from ArXiv.

## Features

- **Automated Paper Collection**: Search and collect papers from ArXiv using targeted keywords
- **Trend Analysis**: Analyze publication trends over time with detailed visualizations
- **Keyword Analysis**: Generate word clouds and frequency analysis of research topics
- **Relevance Scoring**: Calculate relevance scores based on keyword matching
- **Data Export**: Export results to CSV, Excel, and markdown reports
- **Comprehensive Visualization**: Generate publication trend plots and statistical charts

## Supported Research Areas

- **VLA (Vision-Language-Action)**: Multimodal robotics, vision-language-action models
- **RFM (Robot Foundation Models)**: General-purpose robotics, foundation models for robots
- **Robot Learning**: Imitation learning, reinforcement learning, robot policies
- **Embodied AI**: Physical intelligence, robot transformers (RT-1, RT-2, OpenVLA)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install arxiv matplotlib pandas numpy seaborn wordcloud requests openpyxl xlsxwriter
```

Or using the requirements file:

```bash
pip install -e .
```

## Usage

### Basic Usage

Run the analyzer with default settings:

```bash
python -m arxiv_robotics_analyzer
```

This will:
1. Search for papers from 2020 onwards
2. Analyze trends and generate visualizations
3. Create keyword analysis and word clouds
4. Generate summary reports
5. Export data to multiple formats

### Customization

You can modify the search terms in the `ArxivRoboticsAnalyzer` class:

```python
analyzer = ArxivRoboticsAnalyzer()
# Modify search_terms dictionary as needed
papers = analyzer.search_all_terms(start_year=2020)
```

## Output Files

The analyzer generates several output files:

- `arxiv_robotics_trends.png` - Publication trend visualizations
- `arxiv_robotics_analysis_report.md` - Comprehensive analysis report
- `arxiv_robotics_papers.csv` - Raw paper data in CSV format
- `arxiv_robotics_analysis.xlsx` - Excel file with data and statistics

## Project Structure

```
arxiv-robotics-analyzer/
├── main.py                    # Main analyzer script
├── pyproject.toml            # Project configuration
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
└── outputs/                 # Generated analysis files (created at runtime)
```

## API Rate Limits

The tool includes built-in rate limiting to respect ArXiv's API guidelines:
- 1-second delay between requests
- Maximum 500 results per search term
- Automatic duplicate removal

## Example Analysis Output

### Basic Statistics
- Total Papers: ~500-1000 papers (depending on search criteria)
- Analysis Period: 2020-2024
- Average Relevance Score: Calculated based on keyword matching

### Trend Analysis
- Yearly publication trends
- Category-wise growth patterns
- Monthly submission patterns
- Relevance score distributions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ArXiv for providing open access to research papers
- The Python scientific computing community for excellent libraries
- Robotics research community for advancing the field

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{arxiv_robotics_analyzer,
  title={ArXiv Robotics Research Trend Analyzer},
  author={Tomohiro Motoda},
  year={2025},
  url={https://github.com/TomohiroMotoda/arxiv-robotics-analyzer}
}
```

## Contact

For questions or suggestions, please open an issue on GitHub or contact [your.email@example.com].
