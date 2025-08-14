from datetime import datetime
from collections import Counter
import re
import time
import requests

# Check for necessary library installations
try:
    import arxiv
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from wordcloud import WordCloud
    import xml.etree.ElementTree as ET
    from urllib.parse import quote
    from tqdm import tqdm
except ImportError as e:
    print(f"Required libraries are not installed: {e}")
    print("Please install with:")
    print("pip install arxiv matplotlib pandas seaborn wordcloud openpyxl xlsxwriter")
    exit(1)

class ArxivRoboticsAnalyzer:
    def __init__(self):
        self.papers = []
        self.search_terms = {
            'VLA': [
                'Vision-Language-Action Model',
                'Vision-Language-Action',
                'VLA',
            ],
            # 'Robotics Foundation Model': [
            #     'Robotics Foundation Model',
            #     'RMB',
            #     'Robotic Foundation Model',
            #     'Robot Foundation Model'
            # ],
            # 'Robotic Manipulation': [
            #     'Robotic Manipulation',
            #     'RM',
            #     'Robotics Manipulation',
            #     'Robot Manipulation'
            # ]
        }
    
    def search_arxiv_api(self, query, max_results=1000, start_year=2020):
        """Search papers using ArXiv API"""
        papers = []
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            for paper in search.results():
                # only include papers published from start_year
                if paper.published.year >= start_year:
                    papers.append({
                        'title': paper.title,
                        'authors': [str(author) for author in paper.authors],
                        'published': paper.published,
                        'updated': paper.updated,
                        'summary': paper.summary,
                        'categories': paper.categories,
                        'arxiv_id': paper.entry_id.split('/')[-1],
                        'pdf_url': paper.pdf_url
                    })
                    
        except Exception as e:
            print(f"Search error: {query} - {e}")
            
        return papers
    
    def search_all_terms(self, start_year=2020):
        """Collect papers for all search terms"""
        all_papers = []
        
        for category, terms in self.search_terms.items():
            print(f"\n=== Searching category: {category} ===")
            
            for term in terms:
                print(f"Searching term: {term}")
                query = f'all:"{term}" AND (cat:cs.RO OR cat:cs.AI OR cat:cs.CV OR cat:cs.LG)' # all: すべてのフィールドを検索している．
                papers = self.search_arxiv_api(query, max_results=500, start_year=start_year)
                
                for paper in papers:
                    paper['search_term'] = term
                    paper['category'] = category
                    
                all_papers.extend(papers)
                time.sleep(1)  # API rate limit countermeasure
                
        # Remove duplicates
        unique_papers = {}
        for paper in all_papers:
            arxiv_id = paper['arxiv_id']
            if arxiv_id not in unique_papers:
                unique_papers[arxiv_id] = paper
            else:
                # Retain the paper that matches more search terms
                existing = unique_papers[arxiv_id]
                if paper.get('relevance_score', 0) > existing.get('relevance_score', 0):
                    unique_papers[arxiv_id] = paper
        
        self.papers = list(unique_papers.values())
        print(f"\nTotal {len(self.papers)} papers retrieved")
        return self.papers
    
    def get_citation_count_semantic_scholar(self, arxiv_id):
        """Get citation count from Semantic Scholar API"""
        try:
            # Semantic Scholar API endpoint
            url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
            params = {
                'fields': 'citationCount,influentialCitationCount,year,title'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'citation_count': data.get('citationCount', 0),
                    'influential_citation_count': data.get('influentialCitationCount', 0),
                    'semantic_scholar_found': True
                }
            else:
                return {
                    'citation_count': 0,
                    'influential_citation_count': 0,
                    'semantic_scholar_found': False
                }
                
        except Exception as e:
            print(f"Citation lookup error for {arxiv_id}: {e}")
            return {
                'citation_count': 0,
                'influential_citation_count': 0,
                'semantic_scholar_found': False
            }
    
    def fetch_citation_counts(self):
        """Fetch citation counts for all papers"""
        print("Fetching citation counts from Semantic Scholar...")
        
        for paper in tqdm(self.papers, desc="📚 Fetching citations", unit="📄 paper"):
            citation_data = self.get_citation_count_semantic_scholar(paper['arxiv_id'])
            paper.update(citation_data)
            time.sleep(0.1)  # 100ms delay between requests
            
        print("Citation count fetching completed.")
    
    def create_dataframe(self):
        """Create pandas DataFrame from papers"""
        if not self.papers:
            print("No paper data available")
            return None
            
        df_data = []
        for paper in self.papers:
            df_data.append({
                'arxiv_id': paper['arxiv_id'],
                'title': paper['title'],
                'authors': ', '.join(paper['authors'][:3]),  # First 3 authors
                'published_date': paper['published'],
                'year': paper['published'].year,
                'month': paper['published'].month,
                'categories': ', '.join(paper['categories']),
                'search_term': paper.get('search_term', ''),
                'category': paper.get('category', ''),
                'relevance_score': paper.get('relevance_score', 0),
                'abstract_length': len(paper['summary']),
                'num_authors': len(paper['authors']),
                'citation_count': paper.get('citation_count', 0),
                'influential_citation_count': paper.get('influential_citation_count', 0),
                'semantic_scholar_found': paper.get('semantic_scholar_found', False)
            })
        
        self.df = pd.DataFrame(df_data)
        return self.df
    
    def display_paper_list(self, sort_by='published_date', ascending=False, top_n=50, show_citations=True):
        """Display paper list in formatted table"""
        if self.df is None:
            print("No data available. Please run search first.")
            return
        
        # Sort dataframe
        df_sorted = self.df.sort_values(by=sort_by, ascending=ascending).head(top_n)
        
        print(f"\n=== Top {top_n} Papers (sorted by {sort_by}) ===")
        print("=" * 120)
        
        for idx, paper in df_sorted.iterrows():
            print(f"\n[{paper['arxiv_id']}] {paper['title'][:80]}...")
            print(f"Authors: {paper['authors']}")
            print(f"Published: {paper['published_date'].strftime('%Y-%m-%d')} | Category: {paper['category']} | Term: {paper['search_term']}")
            
            if show_citations and paper['semantic_scholar_found']:
                print(f"Citations: {paper['citation_count']} (Influential: {paper['influential_citation_count']})")
            elif show_citations:
                print("Citations: Not available")
            
            print(f"ArXiv URL: https://arxiv.org/abs/{paper['arxiv_id']}")
            print("-" * 120)
    
    def display_top_cited_papers(self, top_n=20):
        """Display most cited papers"""
        if self.df is None:
            return
            
        # Filter papers with citation data
        cited_papers = self.df[self.df['semantic_scholar_found'] == True]
        
        if cited_papers.empty:
            print("No citation data available.")
            return
        
        # Sort by citation count
        top_cited = cited_papers.sort_values('citation_count', ascending=False).head(top_n)
        
        print(f"\n=== Top {top_n} Most Cited Papers ===")
        print("=" * 120)
        
        for idx, paper in top_cited.iterrows():
            print(f"\n{idx+1}. [{paper['arxiv_id']}] {paper['title'][:70]}...")
            print(f"Authors: {paper['authors']}")
            print(f"Published: {paper['published_date'].strftime('%Y-%m-%d')} | Citations: {paper['citation_count']} (Influential: {paper['influential_citation_count']})")
            print(f"Category: {paper['category']} | Search Term: {paper['search_term']}")
            print("-" * 120)
    
    def plot_publication_trends(self, save_path=None):
        """Plot publication trends over time"""
        if self.df is None:
            return
            
        plt.figure(figsize=(15, 12))
        
        # Yearly publication count
        plt.subplot(2, 2, 1)
        yearly_counts = self.df.groupby('year').size()
        plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=8)
        plt.title('Yearly Paper Submission Trends', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Number of Papers')
        plt.grid(True, alpha=0.3)
        
        # Category-wise yearly publication count
        plt.subplot(2, 2, 2)
        category_yearly = self.df.groupby(['year', 'category']).size().unstack(fill_value=0)
        category_yearly.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Category-wise Yearly Paper Counts', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Number of Papers')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        
        # Monthly publication count (last 2 years)
        plt.subplot(2, 2, 3)
        recent_df = self.df[self.df['year'] >= 2023]
        if not recent_df.empty:
            monthly_counts = recent_df.groupby(['year', 'month']).size().reset_index()
            monthly_counts['date'] = pd.to_datetime(monthly_counts[['year', 'month']].assign(day=1))
            plt.plot(monthly_counts['date'], monthly_counts[0], marker='o')
            plt.title('Monthly Paper Submission Trends (From 2023)', fontsize=14, fontweight='bold')
            plt.xlabel('Month')
            plt.ylabel('Number of Papers')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # Relevance score distribution
        plt.subplot(2, 2, 4)
        plt.hist(self.df['relevance_score'], bins=20, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Paper Relevance Scores', fontsize=14, fontweight='bold')
        plt.xlabel('Relevance Score')
        plt.ylabel('Number of Papers')
        plt.grid(True, alpha=0.3)
        
        # Add citation analysis plot if citation data is available
        if 'citation_count' in self.df.columns and self.df['citation_count'].sum() > 0:
            plt.subplot(2, 3, 5)
            cited_papers = self.df[self.df['citation_count'] > 0]
            plt.scatter(cited_papers['year'], cited_papers['citation_count'], alpha=0.6)
            plt.title('Citation Count vs Publication Year', fontsize=14, fontweight='bold')
            plt.xlabel('Publication Year')
            plt.ylabel('Citation Count')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 3, 6)
            plt.hist(self.df['citation_count'], bins=20, alpha=0.7, edgecolor='black')
            plt.title('Citation Count Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Citation Count')
            plt.ylabel('Number of Papers')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_keywords(self, top_n=20):
        """Analyze keywords and generate word cloud"""
        # Combine all text
        all_text = ' '.join([paper['title'] + ' ' + paper['summary'] for paper in self.papers])
        
        # Remove common words
        stop_words = {'the', 'and', 'from', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'we', 'they', 'our', 'their'}
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        words = [word for word in words if word not in stop_words]
        
        # Frequent words
        word_counts = Counter(words)
        top_words = word_counts.most_common(top_n)
        
        print(f"\n=== Top {top_n} Frequent Keywords ===")
        for word, count in top_words:
            print(f"{word}: {count}")
        
        # Generate word cloud
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(all_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Keyword Word Cloud from Papers', fontsize=14, fontweight='bold')
        
        plt.subplot(1, 2, 2)
        words, counts = zip(*top_words[:15])
        plt.barh(range(len(words)), counts)
        plt.yticks(range(len(words)), words)
        plt.xlabel('Frequency')
        plt.title('Top 15 Frequent Keywords', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return top_words
    
    def generate_summary_report(self, output_path=None):
        """Generate summary report"""
        if self.df is None:
            return
            
        report = []
        report.append("# ArXiv Robotics Foundation Model & VLA Research Trend Analysis Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("## Basic Statistics")
        report.append(f"- Total Papers: {len(self.df)}")
        report.append(f"- Analysis Period: {self.df['year'].min()} - {self.df['year'].max()}")
        report.append(f"- Average Relevance Score: {self.df['relevance_score'].mean():.2f}")
        
        # Add citation statistics if available
        if 'citation_count' in self.df.columns:
            cited_papers = self.df[self.df['semantic_scholar_found'] == True]
            report.append(f"- Papers with Citation Data: {len(cited_papers)}")
            if len(cited_papers) > 0:
                report.append(f"- Average Citations: {cited_papers['citation_count'].mean():.2f}")
                report.append(f"- Total Citations: {cited_papers['citation_count'].sum()}")
        
        report.append("\n## Yearly Submission Trends")
        yearly_stats = self.df.groupby('year').size()
        for year, count in yearly_stats.items():
            report.append(f"- {year}: {count} papers")
        
        report.append("\n## Category-wise Statistics")
        category_stats = self.df.groupby('category').size().sort_values(ascending=False)
        for category, count in category_stats.items():
            report.append(f"- {category}: {count} papers")
        
        report.append("\n## Highly Relevant Papers (Score >= 5)")
        high_relevance = self.df[self.df['relevance_score'] >= 5].sort_values('relevance_score', ascending=False)
        for _, paper in high_relevance.head(10).iterrows():
            report.append(f"- [{paper['arxiv_id']}] {paper['title']} (Score: {paper['relevance_score']})")
        
        # Add most cited papers section
        if 'citation_count' in self.df.columns and self.df['citation_count'].sum() > 0:
            report.append("\n## Most Cited Papers (Top 10)")
            top_cited = self.df[self.df['citation_count'] > 0].sort_values('citation_count', ascending=False).head(10)
            for _, paper in top_cited.iterrows():
                report.append(f"- [{paper['arxiv_id']}] {paper['title']} ({paper['citation_count']} citations)")
        
        report_text = '\n'.join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved: {output_path}")
        
        print(report_text)
        return report_text
    
    def export_data(self, csv_path=None, excel_path=None):
        """Export data to CSV and Excel"""
        if self.df is None:
            return
            
        if csv_path:
            self.df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"CSV file saved: {csv_path}")
        
        if excel_path:
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                self.df.to_excel(writer, sheet_name='arXiv paper DB', index=False)
                
                # Statistics sheet
                stats_df = pd.DataFrame({
                    'Year': self.df.groupby('year').size(),
                    'Category': self.df.groupby('category').size()
                })
                stats_df.to_excel(writer, sheet_name='統計')
                
            print(f"Excel file saved: {excel_path}")

def main():
    """Main execution function"""
    print("Starting ArXiv Robotics Foundation Model & VLA research trend analysis...")
    
    # Initialize analyzer
    analyzer = ArxivRoboticsAnalyzer()
    
    # Search papers (from 2020)
    print("Searching for papers...")
    papers = analyzer.search_all_terms(start_year=2020)
    
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
    print("Generating plots...")
    analyzer.plot_publication_trends(save_path='arxiv_robotics_trends.png')
    
    # Keyword analysis
    print("Analyzing keywords...")
    top_keywords = analyzer.analyze_keywords(top_n=30)
    
    # Report generation
    print("Generating summary report...")
    analyzer.generate_summary_report(output_path='arxiv_robotics_analysis_report.md')
    
    # Data export (TODO: Data bugs in export)
    print("Exporting data...")
    analyzer.export_data(
        csv_path='arxiv_robotics_papers.csv',
        excel_path='arxiv_robotics_analysis.xlsx'
    )
    
    print("\nAnalysis completed!")
    print("Generated files:")
    print("- arxiv_robotics_trends.png (trend plot)")
    print("- arxiv_robotics_analysis_report.md (analysis report)")
    print("- arxiv_robotics_papers.csv (paper data)")

if __name__ == "__main__":
    # Japanese font settings (optional)
    plt.rcParams['font.family'] = ['DejaVu Sans']
    main()
