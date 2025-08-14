# 🤖📊 *ArXiv*RoboTrends: A Comprehensive Analysis of Robotics Research Trends on *ArXiv* ($\beta$)

A Python tool for analyzing research trends in robotics foundation models and Vision-Language-Action (VLA) papers from ArXiv.

## TODO

- [x] Update installation instructions and usage command in README.md
- [ ] Customize search terms for specific research areas
- [ ] Output path customization

## Features

- **Automated Paper Search**: Search  papers from ArXiv using targeted keywords
- **Trend Analysis**: Analyze publication trends over time with detailed visualizations
- **Keyword Analysis**: Generate word clouds and frequency analysis of research topics
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
git clone https://github.com/TomohiroMOTODA/arXivSearch.git
cd arXivSearch

pip install -e .
```

## Usage

### Basic Usage

Run the analyzer with default settings:

```bash
python -m arxiv_roboz_trends
```

This will:

1. Search for papers from 2020 onwards
2. Analyze trends and generate visualizations
3. Create keyword analysis and word clouds
4. Generate summary reports
5. Export data to multiple formats

### Customization (TBD)

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
- (TBD) `arxiv_robotics_analysis.xlsx` - Excel file with data and statistics

## Project Structure

```markdown
arxiv-robotics-analyzer/
├── src/
│   └── arxiv-robotics-analyzer/   # Main analyzer package
│       ├── lib/
│       │   ├── __init__.py
│       │   └── arxiv_analyzer.py
│       ├── __init__.py
│       └── __main__.py
├── pyproject.toml                 # Project configuration
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
└── results/                       # Generated analysis files (created at runtime)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{arxiv_robotics_analyzer,
  title={{ArXiv}RoboTrends: A Comprehensive Analysis of Robotics Research Trends on {arXiv}},
  author={Tomohiro Motoda},
  year={2025},
  url={https://github.com/TomohiroMOTODA/arXivSearch}
}
```
