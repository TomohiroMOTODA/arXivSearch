import argparse
from .lib import ArxivRoboticsAnalyzer

def main(keywords=None, start_year=2020):
    """Main execution function"""
    print("Starting ArXiv Robotics Foundation Model & VLA research trend analysis...")
    
    # Initialize analyzer
    analyzer = ArxivRoboticsAnalyzer(keywords=keywords)
    
    # Search papers (from 2020)
    print("Searching for papers...")
    papers = analyzer.search_all_terms(start_year=start_year)

    if not papers:
        print("No papers found.")
        return
    
    # Fetch citation counts
    print("Fetching citation counts...")
    analyzer.fetch_citation_counts()
    
    # Create dataframe
    print("Organizing data...")
    df = analyzer.create_dataframe()
    
    # Display paper lists
    print("Displaying paper lists...")
    analyzer.display_paper_list(sort_by='published_date', top_n=30)
    analyzer.display_top_cited_papers(top_n=15)
    
    # Visualize results
    print("Generating quarterly publication trends...")
    analyzer.plot_publication_trends(
        save_path='./results/arxiv_robotics_trends.png',
        csv_path='./results/quarterly_trends.csv',
        tikz_path='./results/quarterly_trends.tex'
    )
    analyzer.create_citation_analysis_plot(save_path='./results/arxiv_citation_analysis.png')
    
    # Keyword analysis
    print("Analyzing keywords...")
    # top_keywords = analyzer.analyze_keywords(top_n=30)  # TODO: Implement this method
    
    # Report generation
    print("Generating summary report...")
    # analyzer.generate_summary_report(output_path='arxiv_robotics_analysis_report.md')  # TODO: Implement this method
    
    # Data export
    print("Exporting data...")
    analyzer.export_data(
        csv_path='./results/arxiv_robotics_papers.csv',
        excel_path='./results/arxiv_robotics_analysis.xlsx'
    )
    
    print("\nAnalysis completed!")
    print("Generated files:")
    print("- arxiv_robotics_trends.png (trend plot)")
    print("- arxiv_citation_analysis.png (citation analysis)")
    print("- arxiv_robotics_papers.csv (paper data)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArXiv Robotics Foundation Model & VLA research trend analysis')
    parser.add_argument('--keywords', nargs='+', help='Keywords to search for (space-separated)')
    parser.add_argument('--start-year', type=int, default=2020, help='Start year for search (default: 2020)')
    args = parser.parse_args()

    if args.keywords:
        print(f"Custom keywords specified: {args.keywords}")
    main(args.keywords, args.start_year)
